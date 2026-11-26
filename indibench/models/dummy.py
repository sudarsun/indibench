"""Deterministic fake model for dry runs, CI and pipeline debugging (no GPU needed).

``model.extra.mode``:
  * ``random``  – answers a random option letter / echoes a random word (default)
  * ``oracle``  – returns the gold answer when the task passes it via ``_gold`` (metric sanity check)
  * ``echo``    – echoes the last user text
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Tuple

from indibench.config import GenerationConfig, ModelConfig
from indibench.models.base import BaseModel, Request


class DummyModel(BaseModel):
    supports_images = True
    supports_loglikelihood = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.mode = (cfg.extra or {}).get("mode", "random")

    @property
    def name(self) -> str:
        return f"dummy-{self.mode}"

    def _rng(self, req: Request) -> random.Random:
        text = "".join(p.get("text", "") for m in req.messages for p in m["content"])
        return random.Random(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))

    def generate(self, requests: List[Request], gen: GenerationConfig) -> List[str]:
        outs = []
        for r in requests:
            last = "".join(p.get("text", "") for p in r.messages[-1]["content"] if p["type"] == "text")
            if self.mode == "oracle" and getattr(r, "_gold", None) is not None:
                outs.append(str(r._gold))  # type: ignore[attr-defined]
            elif self.mode == "echo":
                outs.append(last[-200:])
            else:
                rng = self._rng(r)
                words = last.split() or ["?"]
                outs.append(f"Answer: {rng.choice('ABCD')}" if "A." in last or "(A)" in last else rng.choice(words))
        return outs

    def generate_ex(self, requests: List[Request], gen: GenerationConfig):
        # extra.needs_tokens: N simulates answers that need N tokens (truncated below that budget)
        need = (self.cfg.extra or {}).get("needs_tokens")
        texts = self.generate(requests, gen)
        return [(t, bool(need) and int(r.max_new_tokens or 0) < int(need)) for t, r in zip(texts, requests)]

    def loglikelihood(self, requests: List[Request]) -> List[List[Tuple[float, int]]]:
        out = []
        for r in requests:
            rng = self._rng(r)
            out.append([(-rng.random() * 5, 1) for _ in r.continuations or []])
        return out
