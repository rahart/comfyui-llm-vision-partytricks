import base64
import io
import json
import os
import re
import socket
import threading
from datetime import datetime

import requests
import numpy as np
from PIL import Image

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    _OUTPUT_DIR = os.getcwd()

try:
    from comfy.model_management import processing_interrupted, throw_exception_if_processing_interrupted
except Exception:
    def processing_interrupted():
        return False

    def throw_exception_if_processing_interrupted():
        pass


def _strip_thinking(text):
    """Remove all known thinking tag formats from text."""
    # Closed tags: <think>, <thinking>, <|think|>, <|thinking|> and their closing variants
    text = re.sub(
        r"<\|?think(?:ing)?\|?>.*?<\|?/think(?:ing)?\|?>",
        "", text, flags=re.DOTALL,
    )
    # Stray closing tags (model emitted extra </think>, or the opening tag was in the
    # chat template rather than the output): everything up to the last one is thinking
    text = re.sub(r"^.*<\|?/think(?:ing)?\|?>", "", text, flags=re.DOTALL)
    # Unclosed tags (model hit max_tokens mid-think)
    text = re.sub(r"<\|?think(?:ing)?\|?>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_response(data, enable_thinking, filter_thinking):
    """Extract response text from API response, handling thinking in multiple formats."""
    message = data["choices"][0]["message"]

    # Some servers (vLLM, llama.cpp) put thinking in a separate field
    thinking_text = message.get("reasoning_content") or message.get("thinking") or ""
    content = message.get("content") or ""

    # If content is empty but thinking exists, the model put everything in thinking
    if not content.strip() and thinking_text.strip():
        raw_text = f"<think>{thinking_text}</think>"
    elif thinking_text:
        raw_text = f"<think>{thinking_text}</think>\n{content}"
    else:
        raw_text = content

    # Filter even with thinking off: some templates (e.g. Muse) always reason.
    if filter_thinking:
        filtered = _strip_thinking(raw_text)
        if not filtered and thinking_text.strip():
            print("[LocalLLM] model returned only reasoning (likely hit max_tokens mid-think); "
                  "response is empty, see raw_response")
    else:
        filtered = raw_text

    return filtered, raw_text


def _tensor_to_base64(tensor):
    """Convert a single image tensor (H, W, C) to base64 PNG."""
    img_np = np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _image_blocks(image, image_detail, max_frames):
    """Turn an IMAGE batch into image_url content blocks, in batch order.

    Batches longer than max_frames (video) are sampled evenly.
    """
    if image is None:
        return []
    batch_size = image.shape[0]
    if batch_size > max_frames:
        indices = np.linspace(0, batch_size - 1, max_frames, dtype=int)
    else:
        indices = range(batch_size)
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{_tensor_to_base64(image[idx])}",
                "detail": image_detail,
            },
        }
        for idx in indices
    ]


def _base_messages(conversation, system_prompt):
    """Start a message list from an optional threaded conversation.

    Copies the incoming history so we never mutate the wire's upstream value. A
    non-empty ``system_prompt`` replaces a leading system turn (or is inserted if
    absent), which is what lets a downstream node refine the system prompt while
    still carrying the prior turns.
    """
    if conversation:
        messages = [dict(m) for m in conversation]
        if system_prompt:
            sys_turn = {"role": "system", "content": system_prompt}
            if messages and messages[0].get("role") == "system":
                messages[0] = sys_turn
            else:
                messages.insert(0, sys_turn)
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    return messages


def _apply_sampling(payload, min_p, top_p, top_k, repeat_penalty,
                    presence_penalty, seed, enable_thinking, thinking_budget,
                    reasoning_effort="off"):
    """Attach optional sampling params to a chat-completions payload in place."""
    if min_p > 0:
        payload["min_p"] = min_p
    # -1 leaves the server's default; 1.0 / 0 are sent so they can switch those defaults off.
    if top_p >= 0:
        payload["top_p"] = top_p
    if top_k >= 0:
        payload["top_k"] = top_k
    if repeat_penalty != 1.0:
        payload["repeat_penalty"] = repeat_penalty
    if presence_penalty != 0.0:
        payload["presence_penalty"] = presence_penalty
    if seed >= 0:
        payload["seed"] = seed

    # A reasoning effort implies thinking; templates skip it when enable_thinking is false.
    thinking = enable_thinking or reasoning_effort != "off"
    payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    if thinking and thinking_budget > 0:
        # llama.cpp field name; 0 leaves thinking unlimited.
        payload["thinking_budget_tokens"] = thinking_budget

    # vLLM reads the top-level field; llama.cpp only forwards chat_template_kwargs to the template.
    if reasoning_effort != "off":
        payload["reasoning_effort"] = reasoning_effort
        payload["chat_template_kwargs"]["reasoning_effort"] = reasoning_effort
    # Muse/Onyx templates read reasoning_strength, default to "high" and can't disable
    # reasoning, so "off" maps to their lowest level. Other templates ignore the key.
    payload["chat_template_kwargs"]["reasoning_strength"] = (
        "minimal" if reasoning_effort == "off" else reasoning_effort)


def _dump_payload(payload, api_url):
    """Write the exact outgoing request to output/llm_debug/last_payload.json.

    Long base64 image URLs are shortened so the file stays readable; the system
    prompt and full history are written verbatim, plus a per-message character
    count so any front-cut-off is obvious at a glance. This shows what the node
    SENDS — compare it to what your server logs to see where truncation happens.
    """
    import copy
    import json

    dbg_dir = os.path.join(_OUTPUT_DIR, "llm_debug")
    os.makedirs(dbg_dir, exist_ok=True)

    safe = copy.deepcopy(payload)
    for msg in safe.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:") and len(url) > 120:
                    block["image_url"]["url"] = url[:80] + f"...<{len(url)} chars>"

    summary = []
    for msg in payload.get("messages", []):
        c = msg.get("content")
        n = len(c) if isinstance(c, str) else sum(
            len(b.get("text", "")) for b in c if isinstance(b, dict))
        summary.append({"role": msg.get("role"), "chars": n})

    out = os.path.join(dbg_dir, "last_payload.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"api_url": api_url, "message_summary": summary, "payload": safe},
                  f, indent=2, ensure_ascii=False)
    return out


def _abort_stream(response):
    """Hard-close the socket so the server sees the disconnect and frees the slot."""
    try:
        response.raw.connection.sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    response.close()


def _post_chat(api_url, payload, timeout, api_key=""):
    """POST a streamed chat request that ComfyUI's cancel button can abort.

    Streaming is what makes cancel reach the server: llama.cpp and vLLM stop
    generating when a streaming client disconnects, while a non-streamed request
    runs to completion server-side. Returns a dict shaped like a non-streamed
    response so _extract_response works unchanged.

    The key comes from the api_key socket, else the LLM_API_KEY env var. It is
    never a widget, so it can't end up in saved workflows or PNG metadata.
    """
    headers = {"Content-Type": "application/json"}
    key = (api_key or "").strip() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = requests.post(api_url, json=payload, headers=headers,
                             stream=True, timeout=timeout)
    response.raise_for_status()

    content, reasoning, errors = [], [], []

    def read():
        try:
            # Decode ourselves: SSE responses carry no charset, so requests would assume Latin-1.
            for raw_line in response.iter_lines():
                line = raw_line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                if "error" in data:
                    raise RuntimeError(f"LLM server error: {data['error']}")
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or delta.get("reasoning")
                                 or delta.get("thinking") or "")
        except Exception as e:
            errors.append(e)

    # Read on a worker so we can poll for cancel even while the server is silent
    # (e.g. processing a long prompt before the first token).
    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.25)
        if processing_interrupted():
            _abort_stream(response)
            worker.join(5)
            throw_exception_if_processing_interrupted()

    if errors:
        raise errors[0]
    return {"choices": [{"message": {"content": "".join(content),
                                     "reasoning_content": "".join(reasoning)}}]}


class LocalLLMQuery:
    """Query a local LLM via OpenAI-compatible API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "api_url": ("STRING", {"default": "http://127.0.0.1:5000/v1/chat/completions"}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072, "step": 1}),
                "timeout": ("INT", {"default": 120, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "conversation": ("LLM_CONVERSATION",),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Explicit control_after_generate stops the frontend migrateWidgetsValues from dropping a saved value.
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1, "control_after_generate": True}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "thinking_budget": ("INT", {"default": 4096, "min": 0, "max": 131072, "step": 256}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
                "debug_dump": ("BOOLEAN", {"default": False}),
                "reasoning_effort": (["off", "minimal", "low", "medium", "high", "xhigh", "max"], {"default": "off"}),
                # Socket-only input; keep last so it can never shift saved widget values.
                "prompt_prefix": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                # Socket-only so the key never lands in workflow JSON; falls back to LLM_API_KEY.
                "api_key": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLM_CONVERSATION")
    RETURN_NAMES = ("response", "raw_response", "conversation")
    FUNCTION = "query"
    CATEGORY = "LLM Vision Party Tricks"

    def query(self, prompt, api_url, model, temperature, max_tokens, timeout,
              conversation=None, system_prompt="", min_p=0.0, top_p=-1.0, top_k=-1,
              repeat_penalty=1.0, presence_penalty=0.0, seed=-1, enable_thinking=False,
              thinking_budget=4096, filter_thinking=True, prompt_prefix="", debug_dump=False, reasoning_effort="off",
              api_key=""):

        if prompt_prefix:
            prompt = prompt_prefix + "\n" + prompt

        messages = _base_messages(conversation, system_prompt)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if model:
            payload["model"] = model
        _apply_sampling(payload, min_p, top_p, top_k, repeat_penalty,
                        presence_penalty, seed, enable_thinking, thinking_budget,
                        reasoning_effort)

        if debug_dump:
            print(f"[LocalLLMQuery] payload dumped to {_dump_payload(payload, api_url)}")

        data = _post_chat(api_url, payload, timeout, api_key)

        filtered, raw_text = _extract_response(data, enable_thinking or reasoning_effort != "off", filter_thinking)

        # Thread the (thinking-stripped) assistant turn back out so history stays clean.
        updated = messages + [{"role": "assistant", "content": filtered}]
        return (filtered, raw_text, updated)


class LocalLLMVisionQuery:
    """Query a local LLM with image/video input via OpenAI-compatible vision API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "api_url": ("STRING", {"default": "http://127.0.0.1:5000/v1/chat/completions"}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072, "step": 1}),
                "timeout": ("INT", {"default": 120, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "conversation": ("LLM_CONVERSATION",),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Explicit control_after_generate stops the frontend migrateWidgetsValues from dropping a saved value.
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1, "control_after_generate": True}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "thinking_budget": ("INT", {"default": 4096, "min": 0, "max": 131072, "step": 256}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
                "video_max_frames": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "keep_image_in_history": ("BOOLEAN", {"default": False}),
                "debug_dump": ("BOOLEAN", {"default": False}),
                "reasoning_effort": (["off", "minimal", "low", "medium", "high", "xhigh", "max"], {"default": "off"}),
                # Socket-only input; keep last so it can never shift saved widget values.
                "prompt_prefix": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                # Socket-only so the key never lands in workflow JSON; falls back to LLM_API_KEY.
                "api_key": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLM_CONVERSATION")
    RETURN_NAMES = ("response", "raw_response", "conversation")
    FUNCTION = "query"
    CATEGORY = "LLM Vision Party Tricks"

    def query(self, prompt, api_url, model, temperature, max_tokens, timeout,
              image=None, conversation=None, system_prompt="", min_p=0.0, top_p=-1.0,
              top_k=-1, repeat_penalty=1.0, presence_penalty=0.0, seed=-1,
              enable_thinking=False, thinking_budget=4096, filter_thinking=True,
              image_detail="auto", video_max_frames=16, keep_image_in_history=False,
              prompt_prefix="", debug_dump=False, reasoning_effort="off", api_key=""):

        if prompt_prefix:
            prompt = prompt_prefix + "\n" + prompt

        messages = _base_messages(conversation, system_prompt)

        # Build user message content
        content = _image_blocks(image, image_detail, video_max_frames)
        content.append({"type": "text", "text": prompt})

        # Send the full content (with images) to the API, but keep the threaded
        # history text-only by default so a downstream text model doesn't choke
        # on image blocks — and so the conversation wire stays small.
        api_messages = messages + [{"role": "user", "content": content}]

        payload = {
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if model:
            payload["model"] = model
        _apply_sampling(payload, min_p, top_p, top_k, repeat_penalty,
                        presence_penalty, seed, enable_thinking, thinking_budget,
                        reasoning_effort)

        if debug_dump:
            print(f"[LocalLLMVisionQuery] payload dumped to {_dump_payload(payload, api_url)}")

        data = _post_chat(api_url, payload, timeout, api_key)

        filtered, raw_text = _extract_response(data, enable_thinking or reasoning_effort != "off", filter_thinking)

        if keep_image_in_history and image is not None:
            hist_user = {"role": "user", "content": content}
        else:
            n_imgs = 0 if image is None else image.shape[0]
            note = f"[{n_imgs} image(s) omitted from history]\n" if n_imgs else ""
            hist_user = {"role": "user", "content": note + prompt}

        updated = messages + [hist_user, {"role": "assistant", "content": filtered}]
        return (filtered, raw_text, updated)


class LLMFewShotExample:
    """Append a worked example to a conversation without calling the model.

    Adds a user turn (image(s) + question) followed by an assistant turn whose
    text YOU write. Chain several of these into a Vision Query: the model reads
    the written answers as its own earlier replies and copies their format.
    Images stay in the history (that is the point of an example).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"default": "What pose is this?", "multiline": True}),
                "answer": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "conversation": ("LLM_CONVERSATION",),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
                "video_max_frames": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("LLM_CONVERSATION",)
    RETURN_NAMES = ("conversation",)
    FUNCTION = "add"
    CATEGORY = "LLM Vision Party Tricks"

    def add(self, question, answer, image=None, conversation=None, system_prompt="",
            image_detail="auto", video_max_frames=16):
        if not answer.strip():
            raise ValueError("LLM Few-Shot Example: answer is empty. "
                             "An example needs the reply you want the model to imitate.")

        messages = _base_messages(conversation, system_prompt)
        blocks = _image_blocks(image, image_detail, video_max_frames)
        if blocks:
            user_turn = {"role": "user", "content": blocks + [{"type": "text", "text": question}]}
        else:
            user_turn = {"role": "user", "content": question}

        return (messages + [user_turn, {"role": "assistant", "content": answer}],)


class LLMSavePromptLog:
    """Append a prompt/response to a log file so good results persist across sessions.

    The in-context conversation wire keeps history only for the current queued
    run; this node writes the text out to disk so winning prompts survive after
    the context clears.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
                "filename": ("STRING", {"default": "prompt_log.txt"}),
            },
            "optional": {
                "label": ("STRING", {"default": ""}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LLM Vision Party Tricks"

    def save(self, text, filename, label="", add_timestamp=True, enabled=True):
        if not enabled:
            return (text,)

        # Resolve relative paths under ComfyUI's output directory.
        path = filename if os.path.isabs(filename) else os.path.join(_OUTPUT_DIR, filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        header_bits = []
        if add_timestamp:
            header_bits.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if label:
            header_bits.append(label)
        header = " | ".join(header_bits)

        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            if header:
                f.write(header + "\n" + "-" * 60 + "\n")
            f.write(text.rstrip() + "\n")

        return (text,)


class LLMConversationToText:
    """Render an LLM_CONVERSATION wire as readable text (for a Show Text node)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conversation": ("LLM_CONVERSATION",),
            },
            "optional": {
                "include_system": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "render"
    CATEGORY = "LLM Vision Party Tricks"

    @staticmethod
    def _content_to_text(content):
        if isinstance(content, str):
            return content
        # Vision turns store a list of typed blocks.
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)

    def render(self, conversation, include_system=True):
        if not conversation:
            return ("",)
        lines = []
        for turn in conversation:
            role = turn.get("role", "?")
            if role == "system" and not include_system:
                continue
            lines.append(f"### {role.upper()}\n{self._content_to_text(turn.get('content', ''))}")
        return ("\n\n".join(lines),)


_BOX_RE = re.compile(r"(-?\d+(?:\.\d+)?)" + r"\s*,\s*(-?\d+(?:\.\d+)?)" * 3)


class QwenNormBBoxToSAM3:
    """Parse x1,y1,x2,y2 boxes (Qwen 0-1000 normalized) into pixel boxes.

    Accepts any four comma-separated numbers anywhere in the text, so bare
    lines, ``[x1, y1, x2, y2]`` and Qwen's ``{"bbox_2d": [...]}`` JSON all work.

    Two outputs, same boxes: ``bboxes`` is the (x1, y1, x2, y2) tuple list that
    comfyui-easy-sam3 reads; ``bounding_boxes`` is core ComfyUI's BOUNDING_BOX
    format ({x, y, width, height} dicts) for the built-in SAM3 Detect node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_string": ("STRING", {"forceInput": True, "multiline": True}),
                "image": ("IMAGE",),
            }
        }

    # New outputs go after existing ones so saved workflow links keep their slots.
    RETURN_TYPES = ("BBOX", "BOUNDING_BOX")
    RETURN_NAMES = ("bboxes", "bounding_boxes")
    FUNCTION = "convert"
    CATEGORY = "LLM Vision Party Tricks"

    def convert(self, bbox_string, image):
        h, w = image.shape[1], image.shape[2]
        boxes = []
        for match in _BOX_RE.finditer(bbox_string):
            x1, y1, x2, y2 = (float(p) for p in match.groups())
            x1, x2 = x1 * w / 1000, x2 * w / 1000
            y1, y2 = y1 * h / 1000, y2 * h / 1000
            x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
            y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        core_boxes = [{"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
                      for x1, y1, x2, y2 in boxes]
        return (boxes, core_boxes)


NODE_CLASS_MAPPINGS = {
    "LocalLLMQuery": LocalLLMQuery,
    "LocalLLMVisionQuery": LocalLLMVisionQuery,
    "LLMFewShotExample": LLMFewShotExample,
    "LLMSavePromptLog": LLMSavePromptLog,
    "LLMConversationToText": LLMConversationToText,
    "QwenNormBBoxToSAM3": QwenNormBBoxToSAM3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLLMQuery": "LLM Query",
    "LocalLLMVisionQuery": "LLM Vision Query",
    "LLMFewShotExample": "LLM Few-Shot Example",
    "LLMSavePromptLog": "LLM Save Prompt Log",
    "LLMConversationToText": "LLM Conversation → Text",
    "QwenNormBBoxToSAM3": "Qwen Norm BBox → SAM3",
}
