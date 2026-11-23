"""GPU capability detection and per-GPU safety adjustments.

Supported targets (single GPU; multi-GPU boxes use device_map / tensor parallelism):

  | GPU             | arch   | CC  | VRAM      | bf16 | FlashAttn-2 | FP8 | profile    |
  |-----------------|--------|-----|-----------|------|-------------|-----|------------|
  | RTX 2080 Ti     | Turing | 7.5 | 11 GB     | no   | no          | no  | rtx2080ti  |
  | RTX 3090        | Ampere | 8.6 | 24 GB     | yes  | yes         | no  | rtx3090    |
  | RTX 4090        | Ada    | 8.9 | 24 GB     | yes  | yes         | yes | rtx4090    |
  | L40S            | Ada    | 8.9 | 48 GB     | yes  | yes         | yes | l40s       |
  | A100 40/80 GB   | Ampere | 8.0 | 40/80 GB  | yes  | yes         | no  | a100-40g / a100-80g |
  | H100 (SXM/PCIe/NVL) | Hopper | 9.0 | 80/94 GB | yes | yes        | yes | h100       |
  | H200 (SXM/NVL)  | Hopper | 9.0 | 141 GB    | yes  | yes         | yes | h200       |

``torch.cuda.is_bf16_supported()`` can report True on Turing because of *emulation*, which
is very slow. This module decides from the compute capability instead.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    name: str
    capability: tuple          # (major, minor)
    memory_gb: float
    count: int = 1

    @property
    def cc(self) -> float:
        return self.capability[0] + self.capability[1] / 10

    @property
    def arch(self) -> str:
        major, minor = self.capability
        if major == 7 and minor >= 5:
            return "Turing"
        if major == 7:
            return "Volta"
        if major == 8 and minor == 9:
            return "Ada"
        if major == 8:
            return "Ampere"
        if major == 9:
            return "Hopper"
        if major >= 10:
            return "Blackwell"
        return "pre-Volta"

    @property
    def bf16(self) -> bool:            # native bf16 tensor cores
        return self.capability >= (8, 0)

    @property
    def flash_attn2(self) -> bool:     # FlashAttention-2 requires Ampere or newer
        return self.capability >= (8, 0)

    @property
    def fp8(self) -> bool:             # FP8 tensor cores: Ada / Hopper / Blackwell
        return self.capability >= (8, 9)

    @property
    def supported_by_torch_cu13(self) -> bool:  # CUDA 13 / PyTorch >= 2.15 floor is sm_75
        return self.capability >= (7, 5)


def detect_gpus() -> List[GPUInfo]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(GPUInfo(p.name, (p.major, p.minor), p.total_memory / 1024**3, torch.cuda.device_count()))
        return out
    except Exception:
        return []


def primary_gpu() -> Optional[GPUInfo]:
    g = detect_gpus()
    return g[0] if g else None


def profile_for(gpu: Optional[GPUInfo]) -> Optional[str]:
    """Map a GPU to a hardware profile name (see ``config.HARDWARE_PROFILES``)."""
    if gpu is None:
        return None
    n = gpu.name.lower()
    if "2080" in n:
        return "rtx2080ti"
    if "3090" in n:
        return "rtx3090"
    if "4090" in n:
        return "rtx4090"
    datacenter = any(k in n for k in ("l40", "a100", "a800", "h100", "h800", "h200", "h20", "gh200"))
    if datacenter and gpu.memory_gb < 30:
        # A MIG slice (e.g. A100 2g.20gb / H100 1g.10gb) exposes only part of the card.
        return "rtx3090" if gpu.memory_gb >= 16 else "rtx2080ti-bf16"
    if "h200" in n or "gh200" in n or (("h100" in n or "h800" in n) and gpu.memory_gb > 100):
        return "h200"
    if "h100" in n or "h800" in n or "h20" in n:
        return "h100"
    if "l40" in n:
        return "l40s"
    if "a100" in n or "a800" in n:
        return "a100-80g" if gpu.memory_gb > 60 else "a100-40g"
    # Unknown card: choose by capability + memory (conservative).
    if not gpu.bf16:
        return "rtx2080ti"            # Turing-class: fp16, no FA2
    if gpu.memory_gb >= 120:
        return "h200"
    if gpu.memory_gb >= 70:
        return "h100" if gpu.fp8 else "a100-80g"
    if gpu.memory_gb >= 44:
        return "l40s"
    if gpu.memory_gb >= 30:
        return "a100-40g"
    return "rtx4090" if gpu.fp8 else "rtx3090"


# ------------------------------------------------------------------ memory estimate
_PARAM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def guess_params_b(model_name: str) -> Optional[float]:
    """Best-effort parameter count from a model id such as ``Qwen2.5-VL-7B-Instruct``.

    MoE ids like ``30B-A3B`` return the *total* size, which is what has to fit in memory.
    """
    hits = [float(x) for x in _PARAM_RE.findall(os.path.basename(model_name.rstrip("/")))]
    hits = [h for h in hits if 0.1 <= h <= 1000]
    return max(hits) if hits else None


def bytes_per_param(dtype: str, quantization: Optional[str]) -> float:
    q = (quantization or "").lower()
    if q in ("4bit", "awq", "gptq", "nf4", "bitsandbytes"):
        return 0.6          # 4-bit weights + scales + unquantized embeddings/vision tower
    if q in ("8bit", "fp8", "int8"):
        return 1.1
    return 4.0 if str(dtype).lower() in ("float32", "fp32") else 2.0


def estimate_weights_gb(params_b: float, dtype: str, quantization: Optional[str]) -> float:
    return params_b * bytes_per_param(dtype, quantization)


@dataclass
class FitReport:
    params_b: Optional[float]
    weights_gb: Optional[float]
    gpu_gb: float
    verdict: str            # ok | tight | no | unknown
    advice: str


def check_fit(model_name: str, dtype: str, quantization: Optional[str], gpu: GPUInfo,
              n_gpus: int = 1) -> FitReport:
    p = guess_params_b(model_name)
    total = gpu.memory_gb * max(1, n_gpus)
    if p is None:
        return FitReport(None, None, total, "unknown", "could not infer parameter count from the model id")
    w = estimate_weights_gb(p, dtype, quantization)
    # Leave room for CUDA context (~0.6 GB), activations, KV cache and visual tokens.
    headroom = 1.5 + 0.08 * total
    if w + headroom <= total * 0.85:
        return FitReport(p, w, total, "ok", "")
    if w + 1.0 <= total * 0.97:
        return FitReport(p, w, total, "tight",
                         "fits, but keep batch small and lower --max-pixels / --max-model-len")
    nxt = ("--quantization 8bit" if bytes_per_param(dtype, quantization) >= 2 and p * 1.1 + headroom <= total * 0.9
           else "--quantization 4bit" if p * 0.6 + headroom <= total * 0.92 else "a smaller model")
    return FitReport(p, w, total, "no", f"~{w:.0f} GB of weights will not fit in {total:.0f} GB; use {nxt}")


# ------------------------------------------------------------------ adjustments
def harden_config(cfg, gpu: Optional[GPUInfo] = None) -> List[str]:
    """Adjust a :class:`BenchConfig` in place so it can run on the detected GPU.

    Returns human-readable notes describing every change/warning (also logged).
    """
    notes: List[str] = []
    mc = cfg.model
    if mc.backend not in ("hf", "vllm"):
        return notes
    gpu = gpu or primary_gpu()
    if gpu is None:
        if mc.backend == "vllm":
            notes.append("No CUDA GPU detected; vLLM needs a GPU.")
        else:
            notes.append("No CUDA GPU detected; the hf backend will run on CPU (very slow).")
        for n in notes:
            log.warning(n)
        return notes

    if not gpu.supported_by_torch_cu13:
        notes.append(f"{gpu.name} (sm_{gpu.capability[0]}{gpu.capability[1]}) is older than Turing: current "
                     "PyTorch (CUDA 13 wheels) has no kernels for it. Install an older PyTorch with cu126 wheels.")

    dt = str(mc.dtype).lower()
    if dt in ("bfloat16", "bf16", "auto") and not gpu.bf16:
        mc.dtype = "float16"
        notes.append(f"{gpu.name} ({gpu.arch}) has no native bfloat16 -> using float16.")
        if any(k in mc.name_or_path.lower() for k in ("gemma", "t5", "mt5")):
            notes.append("This model family is known to overflow (inf/NaN) in float16. If outputs are empty "
                         "or garbage, use a bf16-capable GPU (RTX 3090/4090, A100, L40S) or a different model.")

    if mc.attn_implementation == "flash_attention_2" and not gpu.flash_attn2:
        mc.attn_implementation = "sdpa"
        notes.append(f"FlashAttention-2 needs Ampere or newer; {gpu.name} -> using sdpa.")

    q = (mc.quantization or "").lower()
    if q == "fp8" and not gpu.fp8:
        notes.append(f"FP8 tensor cores need Ada/Hopper (RTX 4090, L40S, H100, H200). On {gpu.name} vLLM can only "
                     "use weight-only FP8 kernels where available; prefer AWQ/GPTQ or --quantization 4bit.")

    if mc.backend == "vllm" and not gpu.bf16:
        notes.append("vLLM on Turing runs in float16 with Triton/FlexAttention fallbacks; some architectures "
                     "fail to start on sm_75. If vLLM errors at startup, use --backend hf.")

    n_gpus = mc.tensor_parallel_size if mc.backend == "vllm" else (gpu.count if mc.device_map == "auto" else 1)
    fit = check_fit(mc.name_or_path, mc.dtype, mc.quantization, gpu, n_gpus)
    if fit.verdict in ("no", "tight"):
        notes.append(f"Memory: {mc.name_or_path} ≈ {fit.params_b:g}B params -> ~{fit.weights_gb:.1f} GB weights "
                     f"on {fit.gpu_gb:.0f} GB: {fit.advice}.")
        if fit.verdict == "no" and mc.backend == "hf" and mc.device_map == "auto":
            notes.append("With device_map=auto, transformers will offload the remainder to CPU RAM "
                         "(works, but 5-20x slower).")

    for n in notes:
        log.warning(n)
    return notes
