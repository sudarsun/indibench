"""vLLM offline backend: highest throughput on a single L40S/A100 (continuous batching, paged KV)."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from indibench.config import GenerationConfig, ModelConfig
from indibench.models.base import BaseModel, Request
from indibench.utils import pil_to_data_uri

log = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    supports_loglikelihood = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover
            raise ImportError("vLLM backend needs: pip install 'indibench[vllm]'") from e
        self.SamplingParams = SamplingParams
        from indibench.hardware import primary_gpu

        gpu = primary_gpu()
        dtype = cfg.dtype
        if gpu is not None and not gpu.bf16 and str(dtype).lower() in ("bfloat16", "bf16", "auto"):
            dtype = "float16"   # Turing (RTX 2080 Ti): no bf16
        kwargs: Dict[str, Any] = {
            "model": cfg.name_or_path,
            "dtype": dtype,
            "trust_remote_code": cfg.trust_remote_code,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "enforce_eager": cfg.enforce_eager,
            "limit_mm_per_prompt": {"image": cfg.image.max_images},
        }
        if cfg.max_model_len:
            kwargs["max_model_len"] = cfg.max_model_len
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        if cfg.processor_name_or_path:
            kwargs["tokenizer"] = cfg.processor_name_or_path
        if cfg.quantization and cfg.quantization not in ("4bit", "8bit"):
            kwargs["quantization"] = cfg.quantization  # awq | gptq | fp8 | bitsandbytes ...
        elif cfg.quantization in ("4bit", "8bit"):
            kwargs["quantization"] = "bitsandbytes"
        mm_kwargs = {}
        if cfg.image.max_pixels:
            mm_kwargs["max_pixels"] = cfg.image.max_pixels
        if cfg.image.min_pixels:
            mm_kwargs["min_pixels"] = cfg.image.min_pixels
        if mm_kwargs and cfg.extra.get("pass_pixel_limits_to_processor", False):
            kwargs["mm_processor_kwargs"] = mm_kwargs
        kwargs.update({k: v for k, v in (cfg.extra or {}).items() if k not in ("pass_pixel_limits_to_processor", "chunk")})
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.supports_images = _vllm_is_multimodal(self.llm)

    # vLLM's scheduler (continuous batching + paged KV cache) only reaches full utilisation with
    # many requests in flight, so hand it large chunks rather than `batch_size` at a time.
    _pil_ok: Optional[bool] = None

    def inference_chunk(self, batch_size: int, modality: str = "text") -> int:
        default = 256 if modality == "image" else 1024
        return max(batch_size, int((self.cfg.extra or {}).get("chunk", default)))

    def _image_part(self, img) -> Dict[str, Any]:
        # Passing the PIL object skips a PNG encode + base64 round-trip per image (CPU-bound).
        if self._pil_ok is not False:
            return {"type": "image_pil", "image_pil": img}
        return {"type": "image_url", "image_url": {"url": pil_to_data_uri(img, "JPEG")}}

    def _to_openai(self, req: Request) -> List[Dict[str, Any]]:
        msgs = []
        for m in req.messages:
            parts = []
            for p in m["content"]:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    parts.append(self._image_part(p["image"]))
            if m["role"] == "system" or not any(p["type"] in ("image_url", "image_pil") for p in parts):
                msgs.append({"role": m["role"], "content": "\n".join(p["text"] for p in parts if p["type"] == "text")})
            else:
                msgs.append({"role": m["role"], "content": parts})
        return msgs

    def _params(self, gen: GenerationConfig, max_new: int, **extra):
        kw: Dict[str, Any] = {"max_tokens": max_new, "temperature": gen.temperature or 0.0, "top_p": gen.top_p}
        if gen.top_k:
            kw["top_k"] = gen.top_k
        if gen.repetition_penalty:
            kw["repetition_penalty"] = gen.repetition_penalty
        if gen.stop:
            kw["stop"] = list(gen.stop)
        kw.update(extra)
        return self.SamplingParams(**kw)

    def generate_ex(self, requests: List[Request], gen: GenerationConfig):
        convs = [self._to_openai(r) for r in requests]
        # one SamplingParams per request, so each answer gets its own token budget
        params = [self._params(gen, int(r.max_new_tokens or 512)) for r in requests]
        chat_kwargs: Dict[str, Any] = {"use_tqdm": False}
        if gen.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": gen.enable_thinking}
        if self.cfg.chat_template:
            try:
                outs = self.llm.chat(convs, params, **chat_kwargs)
                if any(r.images for r in requests):
                    self._pil_ok = True
            except Exception as e:
                if self._pil_ok is None and any(r.images for r in requests) and "image_pil" in str(e).lower() + repr(e):
                    log.info("This vLLM version has no image_pil content parts; sending images as JPEG data URIs")
                    self._pil_ok = False
                    convs = [self._to_openai(r) for r in requests]
                    outs = self.llm.chat(convs, params, **chat_kwargs)
                else:
                    raise
        else:
            prompts = ["\n\n".join(m["content"] if isinstance(m["content"], str) else "" for m in c) for c in convs]
            outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [(o.outputs[0].text, getattr(o.outputs[0], "finish_reason", None) == "length")
                if o.outputs else ("", False) for o in outs]

    def count_tokens(self, texts: List[str]) -> List[int]:
        try:
            return [len(x) for x in self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]]
        except Exception:
            return super().count_tokens(texts)

    def context_length(self) -> Optional[int]:
        try:
            return int(self.llm.llm_engine.model_config.max_model_len)
        except Exception:
            return super().context_length()

    def release_memory(self) -> None:
        # vLLM pre-allocates its KV cache once; only the prefix / multimodal caches can be dropped.
        for name in ("reset_prefix_cache", "reset_mm_cache"):
            fn = getattr(self.llm, name, None)
            if fn:
                try:
                    fn()
                except Exception:
                    pass
        super().release_memory()

    def loglikelihood(self, requests: List[Request]) -> List[List[Tuple[float, int]]]:
        """Score continuations with ``prompt_logprobs`` (text-only prompts)."""
        from vllm.inputs import TokensPrompt

        results = []
        for r in requests:
            if r.images:
                raise NotImplementedError("vLLM log-likelihood with images is not supported; use mcq_mode=generate")
            prompt = self.tokenizer.apply_chat_template(r.text_only_messages(), add_generation_prompt=True, tokenize=False)
            ctx_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompts, lens = [], []
            for c in r.continuations or []:
                full = self.tokenizer(prompt + c, add_special_tokens=False)["input_ids"]
                prompts.append(TokensPrompt(prompt_token_ids=full))
                lens.append(len(full) - len(ctx_ids))
            sp = self.SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
            outs = self.llm.generate(prompts, sp, use_tqdm=False)
            scores = []
            for o, n in zip(outs, lens):
                ids = o.prompt_token_ids
                total = 0.0
                for pos in range(len(ids) - n, len(ids)):
                    lp = o.prompt_logprobs[pos]
                    total += lp[ids[pos]].logprob if lp and ids[pos] in lp else -math.inf
                scores.append((total, n))
            results.append(scores)
        return results

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d.update({"max_model_len": self.cfg.max_model_len, "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
                  "quantization": self.cfg.quantization})
        return d


def _vllm_is_multimodal(llm) -> bool:
    """Ask vLLM whether the loaded architecture accepts images (defaults to True if unknown)."""
    for path in ("llm_engine.model_config", "model_config", "llm_engine.vllm_config.model_config"):
        obj = llm
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        val = getattr(obj, "is_multimodal_model", None)
        if isinstance(val, bool):
            return val
    return True
