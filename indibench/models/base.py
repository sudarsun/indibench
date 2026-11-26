"""Backend-agnostic request/response types and the model interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from indibench.config import GenerationConfig, ModelConfig


@dataclass
class Request:
    """One model call.

    ``messages`` uses an OpenAI-like structure with PIL images inline::

        [{"role": "system", "content": [{"type": "text", "text": "..."}]},
         {"role": "user", "content": [{"type": "image", "image": <PIL>}, {"type": "text", "text": "..."}]}]

    For log-likelihood scoring, ``continuations`` lists candidate assistant replies.
    """

    messages: List[Dict[str, Any]]
    continuations: Optional[List[str]] = None
    max_new_tokens: Optional[int] = None

    @property
    def images(self) -> List[Any]:
        return [p["image"] for m in self.messages for p in m["content"] if p.get("type") == "image"]

    def text_only_messages(self) -> List[Dict[str, Any]]:
        """Collapse content parts to plain strings (for text-only chat templates)."""
        out = []
        for m in self.messages:
            text = "\n".join(p["text"] for p in m["content"] if p.get("type") == "text")
            out.append({"role": m["role"], "content": text})
        return out


class BaseModel:
    supports_images: bool = True
    supports_loglikelihood: bool = False

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name_or_path or self.cfg.backend

    def inference_chunk(self, batch_size: int, modality: str = "text") -> int:
        """How many requests the runner should hand over per call.

        Static-batching backends (transformers) want exactly ``batch_size``; schedulers such as
        vLLM or a remote server want far more in flight to keep the GPU saturated.
        """
        return batch_size

    def reset_capacity(self) -> None:
        """Forget any batch-size limit learned from OOMs (called at the start of each task)."""

    def generate(self, requests: List[Request], gen: GenerationConfig) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests: List[Request]) -> List[List[float]]:
        """Return, per request, the summed log-prob of each continuation."""
        raise NotImplementedError(f"{type(self).__name__} does not support log-likelihood scoring")

    def info(self) -> Dict[str, Any]:
        return {"backend": self.cfg.backend, "name": self.name}

    def close(self) -> None:
        pass


def load_model(cfg: ModelConfig) -> BaseModel:
    model = _load(cfg)
    if cfg.supports_images is not None:      # user override wins over auto-detection
        model.supports_images = bool(cfg.supports_images)
    return model


def _load(cfg: ModelConfig) -> BaseModel:
    b = cfg.backend
    if b == "hf":
        from indibench.models.hf import HFModel

        return HFModel(cfg)
    if b == "vllm":
        from indibench.models.vllm_backend import VLLMModel

        return VLLMModel(cfg)
    if b == "openai":
        from indibench.models.openai_api import OpenAIModel

        return OpenAIModel(cfg)
    if b == "dummy":
        from indibench.models.dummy import DummyModel

        return DummyModel(cfg)
    raise ValueError(f"Unknown backend {b!r}")
