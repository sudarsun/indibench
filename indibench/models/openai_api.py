"""OpenAI-compatible chat-completions backend.

Use it against a local ``vllm serve`` / SGLang / TGI / Ollama / LM Studio server (keeps
the model warm across many benchmark runs) or any hosted OpenAI-compatible API.
Uses only the standard library, so no extra dependency is required.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from indibench.config import GenerationConfig, ModelConfig
from indibench.models.base import BaseModel, Request
from indibench.utils import pil_to_data_uri

log = logging.getLogger(__name__)


def chat_completion(api_base: str, api_key: str, payload: Dict[str, Any], timeout: float, retries: int) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(delay + random.random())
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(delay + random.random())
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"Cannot reach {url}: {e}") from e
    raise RuntimeError("unreachable")


class OpenAIModel(BaseModel):
    supports_images = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.api_key = os.environ.get(cfg.api_key_env, "EMPTY")
        self.model_id = cfg.api_model or cfg.name_or_path

    def _messages(self, req: Request) -> List[Dict[str, Any]]:
        msgs = []
        for m in req.messages:
            has_img = any(p["type"] == "image" for p in m["content"])
            if not has_img:
                msgs.append({"role": m["role"], "content": "\n".join(p["text"] for p in m["content"] if p["type"] == "text")})
                continue
            parts = []
            for p in m["content"]:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                else:
                    parts.append({"type": "image_url", "image_url": {"url": pil_to_data_uri(p["image"], self.cfg.image.format)}})
            msgs.append({"role": m["role"], "content": parts})
        return msgs

    def _one(self, req: Request, gen: GenerationConfig) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": self._messages(req),
            "max_tokens": req.max_new_tokens or gen.max_new_tokens,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
        }
        if gen.stop:
            payload["stop"] = list(gen.stop)[:4]
        if gen.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": gen.enable_thinking}
        payload.update(self.cfg.extra or {})
        resp = chat_completion(self.cfg.api_base, self.api_key, payload, self.cfg.request_timeout, self.cfg.max_retries)
        try:
            msg = resp["choices"][0]["message"]
            return msg.get("content") or ""
        except (KeyError, IndexError):
            log.warning("Malformed API response: %s", str(resp)[:300])
            return ""

    def inference_chunk(self, batch_size: int, modality: str = "text") -> int:
        # Keep the server's scheduler fed: several requests per worker thread in flight.
        return max(batch_size, self.cfg.concurrency * 4)

    def generate(self, requests: List[Request], gen: GenerationConfig) -> List[str]:
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as ex:
            return list(ex.map(lambda r: self._one(r, gen), requests))
