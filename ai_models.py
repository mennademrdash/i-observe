from __future__ import annotations

import config  # noqa: F401  # sets BLAS thread caps before numpy/OpenCV load

import base64
import json
import os
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

DEFAULT_VIDEOMAE_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
DEFAULT_NUM_FRAMES = 16
DEFAULT_FRAME_SIZE = 224


class VideoMAEActionRecognizer:
    """VideoMAE video classification over a clip window.

    The checkpoint is loaded from the Hugging Face cache on first use and kept on disk
    afterwards. Construction raises if the model cannot be loaded, so callers can report
    the integration as unavailable instead of silently producing no action labels.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VIDEOMAE_ID,
        num_frames: int = DEFAULT_NUM_FRAMES,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

        self.torch = torch
        self.num_frames = num_frames
        self.model_id = model_id
        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only
        ).eval()
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}

    def _to_clip(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Uniformly sample/pad to ``num_frames`` RGB frames at the model's resolution."""
        if not frames:
            raise ValueError("VideoMAE needs at least one frame")
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 and f.shape[2] == 3 else f for f in frames]
        if len(rgb) >= self.num_frames:
            idx = np.linspace(0, len(rgb) - 1, self.num_frames).astype(int)
            clip = [rgb[i] for i in idx]
        else:
            clip = list(rgb) + [rgb[-1]] * (self.num_frames - len(rgb))
        size = getattr(self.processor, "size", None)
        target = DEFAULT_FRAME_SIZE
        if isinstance(size, dict):
            target = int(size.get("shortest_edge", size.get("height", DEFAULT_FRAME_SIZE)))
        return [cv2.resize(f, (target, target), interpolation=cv2.INTER_LINEAR) for f in clip]

    def predict(self, frames: list[np.ndarray], top_k: int = 3) -> list[dict]:
        if not frames:
            return []
        clip = self._to_clip(frames)
        # VideoMAEImageProcessor accepts a list of frames as one video and produces
        # pixel_values of shape [1, num_frames, 3, H, W].
        inputs = self.processor(clip, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError(f"VideoMAE processor returned unexpected keys: {list(inputs)}")
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim != 5:
            raise RuntimeError(f"Expected VideoMAE pixel_values [B,T,C,H,W], got {tuple(pixel_values.shape)}")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = logits.softmax(-1)
        k = min(top_k, probs.numel())
        values, indices = probs.topk(k)
        return [
            {"label": self.labels.get(int(i), str(int(i))), "score": float(v)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]


class OpenAICompatibleVLM:
    """Chat-completions client for any OpenAI-compatible server.

    Covers OpenAI, Ollama, vLLM, LM Studio and OpenRouter -- they all expose
    /chat/completions with a Bearer token. Only the base URL, the key source and the
    default model differ, so one client serves every provider.
    """

    def __init__(self, base_url=None, api_key=None, model=None, timeout=180, max_side=768, extra_headers=None):
        self.url = (base_url or os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/") + "/chat/completions"
        self.key = api_key if api_key is not None else os.getenv("VLM_API_KEY", "")
        self.model = model or os.getenv("VLM_MODEL", "qwen2.5vl:7b")
        self.timeout = timeout
        self.max_side = max_side
        self.extra_headers = dict(extra_headers or {})

    @property
    def base(self) -> str:
        return self.url.rsplit("/", 1)[0]

    def ping(self) -> dict:
        """Ask the server what models it has. Used to report availability honestly."""
        models_url = self.base + "/models"
        req = urllib.request.Request(models_url, headers={"Authorization": "Bearer " + self.key})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            names = [m.get("id") for m in data.get("data", [])]
            return {"ok": True, "url": models_url, "models": names, "model": self.model}
        except Exception as exc:
            return {"ok": False, "url": models_url, "error": f"{type(exc).__name__}: {exc}", "model": self.model}

    def _encode(self, path: str) -> str:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > self.max_side:
            scale = self.max_side / longest
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError(f"Cannot re-encode image: {path}")
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def describe(self, image_paths, question):
        if not image_paths:
            return None
        content = [{"type": "text", "text": "Answer only from visible evidence. If uncertain, say so. " + str(question)}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": self._encode(str(path))}})
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 300,
            }
        ).encode()
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.key}
        headers.update(self.extra_headers)
        req = urllib.request.Request(self.url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise RuntimeError(f"VLM HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"VLM request failed: {type(exc).__name__}: {exc}") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"VLM returned no choices: {json.dumps(payload)[:400]}")
        return choices[0]["message"]["content"]


# --------------------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------------------

# Every provider is OpenAI-compatible, so they share one client. Only the base URL, the
# environment variable the key comes from, and the default model differ.
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,  # local server, no key
        "model": "qwen2.5vl:7b",
        "label": "Ollama (local)",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "label": "OpenAI",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        # Vision-capable and available on OpenRouter.
        "model": "google/gemini-2.0-flash-001",
        "label": "OpenRouter",
    },
}


def provider_name() -> str:
    return os.getenv("VLM_PROVIDER", "ollama").strip().lower()


def vlm_status() -> dict:
    """Describe the configured VLM without leaking or requiring the key value."""
    name = provider_name()
    spec = PROVIDERS.get(name)
    if spec is None:
        return {"ok": False, "provider": name, "detail": f"unknown VLM_PROVIDER {name!r}"}
    key_env = spec["key_env"]
    out = {
        "provider": name,
        "label": spec["label"],
        "base_url": os.getenv("VLM_BASE_URL") or spec["base_url"],
        "model": os.getenv("VLM_MODEL") or spec["model"],
    }
    if key_env is None:
        out["key_env"] = None
        out["ok"] = False
        out["detail"] = f"local {spec['label']} server not detected"
        return out
    out["key_env"] = key_env
    out["key_present"] = bool(os.getenv(key_env))
    # ok is deliberately False until a real request succeeds: a present key is not proof
    # of a working model. VLM_LAST_RESULT records the last verified call.
    last = VLM_LAST_RESULT.get(key_env)
    if out["key_present"] and last and last.get("ok"):
        out["ok"] = True
        out["detail"] = (
            f"verified {last.get('elapsed_s')}s ago — “{str(last.get('answer',''))[:90]}”"
        )
    elif out["key_present"]:
        out["ok"] = False
        out["detail"] = (
            f"{key_env} present — not verified yet (press RUN ONE REAL VLM REQUEST)"
        )
    else:
        out["ok"] = False
        out["detail"] = f"{key_env} not set in the process environment"
    return out


def make_vlm() -> OpenAICompatibleVLM:
    """Build the VLM client for the configured provider.

    Raises RuntimeError with an actionable message when the provider is unknown or its
    key is missing. The key is read from the process environment only - never written to
    disk, logs or source.
    """
    name = provider_name()
    spec = PROVIDERS.get(name)
    if spec is None:
        raise RuntimeError(f"Unknown VLM_PROVIDER {name!r}; choose one of {sorted(PROVIDERS)}")
    key_env = spec["key_env"]
    key = os.getenv(key_env) if key_env else ""
    if key_env and not key:
        raise RuntimeError(f"{key_env} is not set in the process environment")
    headers = {}
    if name == "openrouter":
        # OpenRouter recommends attribution headers; both are non-identifying metadata.
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
        headers["X-Title"] = os.getenv("OPENROUTER_SITE_NAME", "I-observe")
    return OpenAICompatibleVLM(
        base_url=os.getenv("VLM_BASE_URL") or spec["base_url"],
        api_key=key,
        model=os.getenv("VLM_MODEL") or spec["model"],
        extra_headers=headers or None,
    )


# Last verified result per key variable, so System Status can distinguish
# "key present" from "provider actually answered".
VLM_LAST_RESULT: dict[str, dict] = {}


def vlm_smoke_test(
    image_path: str,
    question: str = "What is in this image? Answer in one short sentence.",
    language: str | None = None,
) -> dict:
    """One minimal real request, used by the UI to prove the VLM path actually works."""
    client = make_vlm()
    if language:
        # The language instruction must lead and stand alone. Appending it to a prompt
        # that already said "answer in one short sentence" produced a header-only reply.
        question = (
            f"Describe this image in {language}. "
            "Write one natural, conversational sentence as if speaking to a friend. "
            "Use colloquial words and natural pronunciation, not textbook or translated phrasing. "
            "Reply with the sentence only - no headings, no labels, no quotes, no English."
        )
    started = time.time()
    key_env = (PROVIDERS.get(provider_name()) or {}).get("key_env") or provider_name()
    try:
        answer = client.describe([image_path], question)
        result = {
            "ok": True,
            "provider": provider_name(),
            "model": client.model,
            "answer": answer,
            "elapsed_s": round(time.time() - started, 2),
        }
    except Exception as exc:
        result = {
            "ok": False,
            "provider": provider_name(),
            "model": client.model,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "elapsed_s": round(time.time() - started, 2),
        }
    VLM_LAST_RESULT[key_env] = dict(result, at=time.time())
    return result
