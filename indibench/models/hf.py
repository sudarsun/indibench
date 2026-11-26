"""Hugging Face ``transformers`` backend for single-GPU (L40S / A100) evaluation.

Works with most chat VLMs exposed through ``AutoModelForImageTextToText`` (Qwen2/2.5/3-VL,
Gemma-3, LLaVA(-OneVision/-Next), Idefics3/SmolVLM, InternVL-hf, Llama-3.2-Vision, Pixtral,
Aya-Vision, ...) and falls back to ``AutoModelForCausalLM`` for text-only LLMs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from indibench.config import GenerationConfig, ModelConfig
from indibench.models.base import BaseModel, Request

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16",
           "half": "float16", "float32": "float32", "fp32": "float32", "auto": "auto"}


class HFModel(BaseModel):
    supports_loglikelihood = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        try:
            import torch
            import transformers
        except ImportError as e:  # pragma: no cover
            raise ImportError("HF backend needs: pip install 'indibench[hf]'") from e
        self.torch = torch
        self.transformers = transformers

        dtype_name = _DTYPES.get(str(cfg.dtype).lower(), cfg.dtype)
        self.dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        from indibench.hardware import primary_gpu

        self.gpu = primary_gpu()
        # Decide from compute capability: is_bf16_supported() also returns True for the
        # (very slow) emulated bf16 path on Turing cards such as the RTX 2080 Ti.
        if self.gpu is not None and not self.gpu.bf16 and self.dtype in ("auto", torch.bfloat16):
            log.warning("%s has no native bf16; using float16", self.gpu.name)
            self.dtype = torch.float16

        proc_path = cfg.processor_name_or_path or cfg.name_or_path
        common = {"trust_remote_code": cfg.trust_remote_code}
        if cfg.revision:
            common["revision"] = cfg.revision

        self.processor = None
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(proc_path, **common)
        except Exception as e:  # text-only models often have no processor
            log.info("AutoProcessor unavailable (%s); using AutoTokenizer", type(e).__name__)
        if self.processor is None or not hasattr(self.processor, "tokenizer"):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(proc_path, **common)
            if self.processor is not None and not hasattr(self.processor, "image_processor"):
                self.processor = None
        else:
            self.tokenizer = self.processor.tokenizer
        self.supports_images = self.processor is not None and hasattr(self.processor, "image_processor")

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model(common)
        self.model.eval()
        log.info("Loaded %s (%s, images=%s) on %s", cfg.name_or_path, type(self.model).__name__,
                 self.supports_images, getattr(self.model, "device", "?"))

    # ------------------------------------------------------------------ loading
    def _load_model(self, common: Dict[str, Any]):
        t = self.transformers
        cfg = self.cfg
        kwargs: Dict[str, Any] = dict(common)
        # transformers >= 4.56 renamed torch_dtype -> dtype
        major_minor = tuple(int(x) for x in t.__version__.split(".")[:2])
        kwargs["dtype" if major_minor >= (4, 56) else "torch_dtype"] = self.dtype
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map
        if cfg.max_gpu_memory_gb and self.torch.cuda.is_available():
            # Cap per-GPU usage (rest is offloaded to CPU) — useful on 11-24 GB consumer cards.
            kwargs["max_memory"] = {i: f"{cfg.max_gpu_memory_gb}GiB" for i in range(self.torch.cuda.device_count())}
            kwargs["max_memory"]["cpu"] = cfg.extra.get("max_cpu_memory", "64GiB")
        if cfg.quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            if self.dtype != "auto":
                compute = self.dtype
            else:
                compute = self.torch.bfloat16 if (self.gpu is None or self.gpu.bf16) else self.torch.float16
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
                if cfg.quantization == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        elif cfg.quantization:
            log.warning("quantization=%s is ignored by the hf backend (pre-quantized checkpoints load as-is)",
                        cfg.quantization)
        kwargs.update({k: v for k, v in (cfg.extra or {}).items() if k not in ("max_cpu_memory",)})

        classes = []
        if self.supports_images:
            for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
                if hasattr(t, name):
                    classes.append(getattr(t, name))
        classes.append(t.AutoModelForCausalLM)

        attn_options = [cfg.attn_implementation] if cfg.attn_implementation else [None]
        if cfg.attn_implementation and cfg.attn_implementation != "sdpa":
            attn_options.append("sdpa")
        if None not in attn_options:
            attn_options.append(None)

        last_err: Optional[Exception] = None
        for cls in classes:
            for attn in attn_options:
                kw = dict(kwargs)
                if attn:
                    kw["attn_implementation"] = attn
                try:
                    model = cls.from_pretrained(cfg.name_or_path, **kw)
                    if attn != cfg.attn_implementation:
                        log.warning("attn_implementation=%s unavailable; using %s", cfg.attn_implementation, attn)
                    if cls is t.AutoModelForCausalLM and self.supports_images:
                        log.warning("Loaded with AutoModelForCausalLM; image inputs may not be supported")
                    return model
                except (ValueError, ImportError, KeyError, TypeError) as e:
                    last_err = e
                    msg = str(e).lower()
                    if "attn" in msg or "flash" in msg or "sdpa" in msg:
                        continue  # try next attention implementation
                    break  # wrong class -> try the next class
        raise RuntimeError(f"Could not load {cfg.name_or_path}: {last_err}")

    # ------------------------------------------------------------------ prompts
    def _to_hf_messages(self, req: Request, with_images: bool) -> List[Dict[str, Any]]:
        if not with_images:
            return req.text_only_messages()
        out = []
        for m in req.messages:
            parts = []
            for p in m["content"]:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    parts.append({"type": "image"})
            out.append({"role": m["role"], "content": parts})
        return out

    def render(self, req: Request, add_generation_prompt: bool = True,
               enable_thinking: Optional[bool] = None) -> str:
        use_proc = self.supports_images and bool(req.images)
        if not self.cfg.chat_template:
            return "\n\n".join(p["text"] for m in req.messages for p in m["content"] if p["type"] == "text")
        templ = self.processor if use_proc else self.tokenizer
        msgs = self._to_hf_messages(req, use_proc)
        extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        try:
            return templ.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, tokenize=False, **extra)
        except Exception as e:
            # Some templates reject a system role or list-style content: merge and retry.
            if msgs and msgs[0]["role"] == "system":
                sys_msg, rest = msgs[0], msgs[1:]
                sys_text = sys_msg["content"] if isinstance(sys_msg["content"], str) else \
                    "\n".join(p.get("text", "") for p in sys_msg["content"])
                if rest and rest[0]["role"] == "user":
                    if isinstance(rest[0]["content"], str):
                        rest[0] = {"role": "user", "content": sys_text + "\n\n" + rest[0]["content"]}
                    else:
                        rest[0] = {"role": "user", "content": [{"type": "text", "text": sys_text}] + rest[0]["content"]}
                return templ.apply_chat_template(rest, add_generation_prompt=add_generation_prompt,
                                                 tokenize=False, **extra)
            raise e

    def _encode(self, texts: List[str], images: Optional[List[List[Any]]]):
        add_special = not self.cfg.chat_template
        if images and self.supports_images:
            attempts = [images, [im for ims in images for im in ims]]
            last = None
            for imgs in attempts:
                for kw in ({"add_special_tokens": add_special}, {}):
                    try:
                        return self.processor(text=texts, images=imgs, return_tensors="pt", padding=True, **kw)
                    except (TypeError, ValueError) as e:
                        last = e
            raise last  # type: ignore[misc]
        return self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=add_special)

    # ------------------------------------------------------------------ generation
    # ------------------------------------------------------------------ batching helpers
    # `_cap` is the largest batch that fit in memory for the current task. It is learned from
    # CUDA OOMs and reused, so one OOM does not repeat on every following batch. The runner
    # resets it at the start of each task (prompt lengths differ between tasks).
    _cap: Optional[int] = None

    def reset_capacity(self) -> None:
        self._cap = None

    @staticmethod
    def _cost(r: Request) -> int:
        return sum(len(p.get("text", "")) for m in r.messages for p in m["content"]) + 4000 * len(r.images)

    def _split(self, items: List[Any]) -> List[List[Any]]:
        n = self._cap or len(items)
        return [items[i:i + n] for i in range(0, len(items), n)] if items else []

    def _note_oom(self, size: int) -> int:
        new = max(1, size // 2)
        if self._cap is None or new < self._cap:
            log.warning("CUDA OOM at batch=%d -> using batch<=%d for the rest of this task", size, new)
            self._cap = new
        self.release_memory()
        return new

    def release_memory(self) -> None:
        # Static/hybrid KV caches (Gemma-3, some Llama configs) are kept on the model between
        # generate() calls and re-allocated larger when a later task needs longer outputs.
        for attr in ("_cache", "_static_cache"):
            if getattr(self.model, attr, None) is not None:
                try:
                    setattr(self.model, attr, None)
                except Exception:
                    pass
        super().release_memory()

    def count_tokens(self, texts: List[str]) -> List[int]:
        if not texts:
            return []
        return [len(x) for x in self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    def context_length(self) -> Optional[int]:
        native = None
        c = getattr(self.model, "config", None)
        for cfg in (getattr(c, "text_config", None), c):
            v = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
            if isinstance(v, int) and v > 0:
                native = v
                break
        vals = [x for x in (self.cfg.max_model_len, native) if x]
        return int(min(vals)) if vals else None

    # ------------------------------------------------------------------ generation
    def generate_ex(self, requests: List[Request], gen: GenerationConfig) -> List[Tuple[str, bool]]:
        outputs: List[Optional[Tuple[str, bool]]] = [None] * len(requests)
        # Never mix image and text-only prompts in one processor call. Within each group,
        # sort by prompt size so a batch holds similar lengths (less padding = more useful work).
        with_img = sorted((i for i, r in enumerate(requests) if r.images), key=lambda i: self._cost(requests[i]))
        without = sorted((i for i, r in enumerate(requests) if not r.images), key=lambda i: self._cost(requests[i]))
        for group in (with_img, without):
            for sub in self._split(group):
                for i, res in zip(sub, self._generate_safe([requests[i] for i in sub], gen)):
                    outputs[i] = res
        return [o or ("", False) for o in outputs]

    def _generate_safe(self, batch: List[Request], gen: GenerationConfig) -> List[Tuple[str, bool]]:
        torch = self.torch
        try:
            return self._generate_batch(batch, gen)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
        # Retry *outside* the except block: while the exception is being handled its traceback
        # keeps the failed batch's GPU tensors alive, so splitting there would OOM again.
        n = self._note_oom(len(batch))
        out: List[Tuple[str, bool]] = []
        for i in range(0, len(batch), n):
            out.extend(self._generate_safe(batch[i:i + n], gen))
        return out

    def _generate_batch(self, batch: List[Request], gen: GenerationConfig) -> List[str]:
        torch = self.torch
        texts = [self.render(r, enable_thinking=gen.enable_thinking) for r in batch]
        images = [r.images for r in batch] if any(r.images for r in batch) else None
        inputs = self._encode(texts, images).to(self.model.device)
        if "pixel_values" in inputs and self.dtype != "auto":
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        max_new = max(int(r.max_new_tokens or 512) for r in batch)
        kw: Dict[str, Any] = {"max_new_tokens": max_new, "pad_token_id": self.tokenizer.pad_token_id}
        if gen.temperature and gen.temperature > 0:
            kw.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p)
            if gen.top_k:
                kw["top_k"] = gen.top_k
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if gen.repetition_penalty:
            kw["repetition_penalty"] = gen.repetition_penalty
        if gen.stop:
            kw.update(stop_strings=list(gen.stop), tokenizer=self.tokenizer)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kw)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        truncated = self._truncated(new_tokens, max_new)
        del out, inputs
        return [(_cut_at_stop(d, gen.stop), t) for d, t in zip(decoded, truncated)]

    def _truncated(self, new_tokens, max_new: int) -> List[bool]:
        """A row is truncated when it used the whole budget and its last token is not an
        end-of-sequence token (finished rows are padded after their EOS)."""
        n = new_tokens.shape[0]
        if new_tokens.shape[1] < max_new:
            return [False] * n
        stop_ids = set()
        for src in (getattr(self.model, "generation_config", None), self.tokenizer):
            e = getattr(src, "eos_token_id", None)
            stop_ids.update(e if isinstance(e, (list, tuple)) else [e] if e is not None else [])
        if self.tokenizer.pad_token_id is not None:
            stop_ids.add(self.tokenizer.pad_token_id)
        last = new_tokens[:, -1].tolist()
        return [int(t) not in stop_ids for t in last]

    # ------------------------------------------------------------------ log-likelihood
    def loglikelihood(self, requests: List[Request]) -> List[List[Tuple[float, int]]]:
        """Batched scoring of all (prompt + continuation) sequences, right-padded so token
        positions match the unpadded sequence exactly."""
        items = []  # (request index, continuation index, full text, images, n_ctx)
        for ri, r in enumerate(requests):
            prompt = self.render(r)
            imgs = list(r.images) if r.images else None
            n_ctx = self._encode([prompt], [imgs] if imgs else None)["input_ids"].shape[1]
            for ci, cont in enumerate(r.continuations or []):
                items.append((ri, ci, prompt + cont, imgs, n_ctx))
        scores: List[List[Tuple[float, int]]] = [[(float("-inf"), 0)] * len(r.continuations or []) for r in requests]
        with_img = [it for it in items if it[3]]
        without = [it for it in items if not it[3]]
        for group in (with_img, without):
            group.sort(key=lambda it: len(it[2]))
            for sub in self._split(group):
                for (ri, ci, *_), val in zip(sub, self._ll_safe(sub)):
                    scores[ri][ci] = val
        return scores

    def _ll_safe(self, batch) -> List[Tuple[float, int]]:
        torch = self.torch
        try:
            return self._ll_batch(batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
        n = self._note_oom(len(batch))   # outside the except block, see _generate_safe
        out: List[Tuple[float, int]] = []
        for i in range(0, len(batch), n):
            out.extend(self._ll_safe(batch[i:i + n]))
        return out

    def _ll_batch(self, batch) -> List[Tuple[float, int]]:
        torch = self.torch
        texts = [it[2] for it in batch]
        imgs = [it[3] for it in batch] if batch[0][3] else None
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            enc = self._encode(texts, imgs).to(self.model.device)
        finally:
            self.tokenizer.padding_side = old_side
        if "pixel_values" in enc and self.dtype != "auto":
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            logits = self.model(**enc).logits
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        out = []
        for j, it in enumerate(batch):
            n_ctx, L = it[4], int(lengths[j])
            if L <= n_ctx:
                out.append((float("-inf"), 0))
                continue
            logp = torch.log_softmax(logits[j, n_ctx - 1:L - 1].float(), dim=-1)
            tgt = enc["input_ids"][j, n_ctx:L]
            out.append((float(logp.gather(1, tgt[:, None]).sum()), int(tgt.shape[0])))
        del logits, enc
        return out

    def info(self) -> Dict[str, Any]:
        torch = self.torch
        d = super().info()
        d.update({
            "model_class": type(self.model).__name__,
            "dtype": str(getattr(self.model, "dtype", self.dtype)),
            "quantization": self.cfg.quantization,
            "params_b": round(sum(p.numel() for p in self.model.parameters()) / 1e9, 3),
            "supports_images": self.supports_images,
        })
        if self.gpu is not None:
            d["gpu"] = f"{self.gpu.name} ({self.gpu.arch}, sm_{self.gpu.capability[0]}{self.gpu.capability[1]})"
        devmap = getattr(self.model, "hf_device_map", None)
        if devmap and any(str(v) in ("cpu", "disk") for v in devmap.values()):
            d["cpu_offload"] = True
        if torch.cuda.is_available():
            d["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        return d

    def close(self) -> None:
        del self.model
        super().release_memory()


def _cut_at_stop(text: str, stops: List[str]) -> str:
    for s in stops or []:
        idx = text.find(s)
        if idx >= 0:
            text = text[:idx]
    return text
