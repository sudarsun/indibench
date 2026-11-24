"""Configuration: dataclass defaults  <  YAML file  <  CLI flags  <  ``--set key=value`` overrides."""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ImageConfig:
    max_pixels: Optional[int] = 1280 * 28 * 28   # ~1.0 MP; keeps visual tokens bounded on one GPU
    min_pixels: Optional[int] = None
    max_images: int = 8                          # max images per prompt (extra images are dropped)
    format: str = "PNG"                          # encoding used for API backends


@dataclass
class ModelConfig:
    backend: str = "hf"                          # hf | vllm | openai | dummy
    name_or_path: str = ""
    revision: Optional[str] = None
    processor_name_or_path: Optional[str] = None
    dtype: str = "bfloat16"                      # bfloat16 | float16 | float32 | auto
    device_map: str = "auto"
    max_gpu_memory_gb: Optional[float] = None    # hf: cap VRAM per GPU, offload the rest to CPU
    attn_implementation: Optional[str] = "sdpa"  # sdpa | flash_attention_2 | eager
    quantization: Optional[str] = None           # 4bit | 8bit (bitsandbytes, hf) | awq/gptq/fp8 (vllm)
    trust_remote_code: bool = False              # model/processor: allow custom modeling code from the Hub
    chat_template: bool = True                   # apply the model's chat template
    system_prompt: Optional[str] = None
    supports_images: Optional[bool] = None       # null = auto-detect; false = text-only model (skip vision tasks)
    max_model_len: Optional[int] = 8192          # truncation / vLLM context
    gpu_memory_utilization: float = 0.90         # vLLM only
    tensor_parallel_size: int = 1                # vLLM only
    enforce_eager: bool = False                  # vLLM only
    api_base: str = "http://localhost:8000/v1"   # openai backend
    api_key_env: str = "OPENAI_API_KEY"
    api_model: Optional[str] = None              # model id sent to API (defaults to name_or_path)
    concurrency: int = 8                         # openai backend parallel requests
    request_timeout: float = 300.0
    max_retries: int = 5
    image: ImageConfig = field(default_factory=ImageConfig)
    extra: Dict[str, Any] = field(default_factory=dict)   # forwarded to from_pretrained / vllm.LLM


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop: List[str] = field(default_factory=list)
    strip_thinking: bool = True                  # remove <think>...</think> before scoring
    enable_thinking: Optional[bool] = None       # passed to chat templates that support it (Qwen3 etc.)


@dataclass
class JudgeConfig:
    enabled: bool = False
    api_base: str = "http://localhost:8001/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = ""
    max_tokens: int = 16
    concurrency: int = 4


@dataclass
class RunConfig:
    tasks: List[str] = field(default_factory=list)     # task names, glob patterns or group names
    languages: Optional[List[str]] = None              # filter language-expanded tasks, e.g. [ta, hi] or [indic]
    modalities: Optional[List[str]] = None             # [text] for text-only models, [image], or null/[all]
    exclude: List[str] = field(default_factory=list)   # task names, globs, tags or groups to drop
    limit: Optional[Union[int, float]] = None          # per task: int = N samples, float in (0,1] = fraction
    sampling: str = "random"                           # random | first | stratified
    stratify_by: Optional[str] = None                  # column for stratified sampling (default: task's group_by)
    seed: int = 1234
    batch_size: int = 8
    auto_reduce_batch: bool = True                     # halve batch size on CUDA OOM
    num_fewshot: Optional[int] = None                  # override per-task default
    mcq_mode: Optional[str] = None                     # generate | loglikelihood (override per task)
    instruction_lang: str = "en"                       # en | native (use native-language instructions if available)
    output_dir: str = "results"
    run_name: Optional[str] = None
    resume: bool = True
    save_predictions: bool = True
    bootstrap: int = 1000                              # bootstrap resamples for 95% CI (0 disables)
    task_files: List[str] = field(default_factory=list)  # extra YAML task-spec files
    cache_dir: Optional[str] = None                    # HF datasets cache
    hf_token_env: str = "HF_TOKEN"
    trust_remote_code: Optional[bool] = None           # datasets: run Hub loading scripts (needs datasets<4)
    streaming: Optional[bool] = None                   # force streaming dataset loading for all tasks
    max_input_chars: Optional[int] = None              # truncate very long contexts (summarisation)
    log_level: str = "INFO"
    log_samples: int = 2                               # print N rendered prompts per task at DEBUG/INFO


@dataclass
class BenchConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    run: RunConfig = field(default_factory=RunConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    # auto (detect) | none | rtx2080ti | rtx3090 | rtx4090 | l40s | a100-40g | a100-80g | h100 | h200
    hardware_profile: Optional[str] = "auto"
    # Convenience: sets BOTH model.trust_remote_code and run.trust_remote_code unless those are given.
    trust_remote_code: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Hardware presets: sane single-GPU defaults. They only fill values the user did not set.
# --------------------------------------------------------------------------------------
HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "l40s": {  # 48 GB, Ada; bf16 + FlashAttention-2 supported
        "run.batch_size": 8,
        "model.gpu_memory_utilization": 0.90,
        "model.max_model_len": 8192,
        "model.image.max_pixels": 1280 * 28 * 28,
    },
    "a100-40g": {
        "run.batch_size": 6,
        "model.gpu_memory_utilization": 0.88,
        "model.max_model_len": 8192,
        "model.image.max_pixels": 1024 * 28 * 28,
    },
    "a100-80g": {
        "run.batch_size": 16,
        "model.gpu_memory_utilization": 0.92,
        "model.max_model_len": 16384,
        "model.image.max_pixels": 1600 * 28 * 28,
    },
    "h100": {  # 80 GB (SXM/PCIe) or 94 GB (NVL), Hopper (sm_90): bf16, FA2 (FA3 in vLLM), FP8
        "run.batch_size": 16,
        "model.gpu_memory_utilization": 0.92,
        "model.max_model_len": 16384,
        "model.image.max_pixels": 1600 * 28 * 28,
    },
    "h200": {  # 141 GB HBM3e, Hopper (sm_90): room for 70B in 8-bit/FP8 or 32B in bf16 with long context
        "run.batch_size": 24,
        "model.gpu_memory_utilization": 0.92,
        "model.max_model_len": 32768,
        "model.image.max_pixels": 1600 * 28 * 28,
    },
    # ---- consumer cards -------------------------------------------------------------
    "rtx4090": {  # 24 GB, Ada (sm_89): bf16, FlashAttention-2, FP8
        "run.batch_size": 4,
        "model.gpu_memory_utilization": 0.88,
        "model.max_model_len": 8192,
        "model.image.max_pixels": 768 * 28 * 28,
        "model.image.max_images": 6,
    },
    "rtx3090": {  # 24 GB, Ampere (sm_86): bf16, FlashAttention-2, no FP8
        "run.batch_size": 4,
        "model.gpu_memory_utilization": 0.88,
        "model.max_model_len": 8192,
        "model.image.max_pixels": 768 * 28 * 28,
        "model.image.max_images": 6,
    },
    "rtx2080ti": {  # 11 GB, Turing (sm_75): NO bf16, NO FlashAttention-2 -> fp16 + sdpa
        "run.batch_size": 2,
        "model.dtype": "float16",
        "model.attn_implementation": "sdpa",
        "model.gpu_memory_utilization": 0.85,
        "model.max_model_len": 4096,
        "model.image.max_pixels": 448 * 28 * 28,
        "model.image.max_images": 4,
        "generation.max_new_tokens": 192,
    },
}

HARDWARE_PROFILES["rtx2080ti-bf16"] = {  # small MIG slice of an A100/H100: bf16-capable but ~10 GB
    k: v for k, v in HARDWARE_PROFILES["rtx2080ti"].items() if k not in ("model.dtype", "model.attn_implementation")}

PROFILE_ALIASES = {"h100-80g": "h100", "h100-sxm": "h100", "h100-pcie": "h100", "h100-nvl": "h100",
                   "h200-141g": "h200", "h200-sxm": "h200", "h200-nvl": "h200", "gh200": "h200",
                   "l40": "l40s", "l40s-48g": "l40s",
                   "a100": "a100-80g", "a100-80": "a100-80g", "a100-40": "a100-40g",
                   "4090": "rtx4090", "rtx-4090": "rtx4090", "3090": "rtx3090", "rtx-3090": "rtx3090",
                   "2080ti": "rtx2080ti", "2080": "rtx2080ti", "rtx-2080ti": "rtx2080ti", "rtx2080": "rtx2080ti"}


def _coerce(value: str) -> Any:
    """Parse a CLI ``--set`` value with YAML semantics (ints, floats, bools, lists, null)."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _get_path(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "extra":
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(cls, data: Dict[str, Any]):
    """Build nested dataclasses from a dict, rejecting unknown keys with a helpful message."""
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in (data or {}).items():
        if k not in known:
            raise ValueError(f"Unknown config key '{k}' for section {cls.__name__}. "
                             f"Valid keys: {sorted(known)}")
        f = known[k]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore
        if is_dataclass(default) and isinstance(v, dict):
            kwargs[k] = _from_dict(type(default), v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def detect_hardware_profile() -> Optional[str]:
    from indibench.hardware import primary_gpu, profile_for

    return profile_for(primary_gpu())


def load_config(
    yaml_path: Optional[str] = None,
    cli_values: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
) -> BenchConfig:
    """Resolve the final configuration.

    ``cli_values`` maps dotted keys (``run.limit``) to values from first-class CLI flags
    (``None`` means "not given"). ``overrides`` are raw ``key=value`` strings from ``--set``.
    """
    data: Dict[str, Any] = BenchConfig().to_dict()
    user: Dict[str, Any] = {}
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")
        user = _deep_merge(user, loaded)
    for k, v in (cli_values or {}).items():
        if v is not None:
            _set_path(user, k, v)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_path(user, k.strip(), _coerce(v.strip()))

    # Hardware profile only fills keys the user did not specify explicitly.
    profile = user.get("hardware_profile", BenchConfig().hardware_profile)
    if isinstance(profile, str):
        profile = PROFILE_ALIASES.get(profile.lower(), profile.lower())
        user["hardware_profile"] = profile
    if profile in ("none", "off"):
        profile = None
        user["hardware_profile"] = None
    backend = _get_path(user, "model.backend") or ModelConfig().backend
    if profile == "auto" and backend in ("openai", "dummy"):
        profile = None          # the GPU (if any) is not ours to size
        user["hardware_profile"] = None
    if profile == "auto":
        profile = detect_hardware_profile()
        user["hardware_profile"] = profile
    if profile:
        if profile not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware_profile {profile!r}; choose from {sorted(HARDWARE_PROFILES)} or 'auto'")
        for k, v in HARDWARE_PROFILES[profile].items():
            if _get_path(user, k) is None:
                _set_path(user, k, v)

    top_trc = user.get("trust_remote_code")
    if top_trc is not None:
        if not isinstance(top_trc, bool):
            raise ValueError("trust_remote_code must be true or false")
        if _get_path(user, "model.trust_remote_code") is None:
            _set_path(user, "model.trust_remote_code", top_trc)
        if _get_path(user, "run.trust_remote_code") is None:
            _set_path(user, "run.trust_remote_code", top_trc)

    data = _deep_merge(data, user)
    cfg = _from_dict(BenchConfig, data)
    _validate(cfg)
    return cfg


def _validate(cfg: BenchConfig) -> None:
    if cfg.model.backend not in ("hf", "vllm", "openai", "dummy"):
        raise ValueError(f"model.backend must be hf|vllm|openai|dummy, got {cfg.model.backend!r}")
    if cfg.model.backend != "dummy" and not cfg.model.name_or_path:
        raise ValueError("model.name_or_path is required (use --model)")
    lim = cfg.run.limit
    if lim is not None:
        if isinstance(lim, bool) or not isinstance(lim, (int, float)):
            raise ValueError("run.limit must be an int (count) or float in (0, 1] (fraction)")
        if isinstance(lim, float) and not (0 < lim <= 1):
            raise ValueError("run.limit as a float must be a fraction in (0, 1]")
        if isinstance(lim, int) and lim <= 0:
            raise ValueError("run.limit must be positive")
    if cfg.run.sampling not in ("random", "first", "stratified"):
        raise ValueError("run.sampling must be random|first|stratified")
    if cfg.run.mcq_mode not in (None, "generate", "loglikelihood"):
        raise ValueError("run.mcq_mode must be generate|loglikelihood")
    if cfg.run.instruction_lang not in ("en", "native"):
        raise ValueError("run.instruction_lang must be en|native")
    if cfg.run.batch_size < 1:
        raise ValueError("run.batch_size must be >= 1")
    if isinstance(cfg.run.tasks, str):
        cfg.run.tasks = [t.strip() for t in cfg.run.tasks.split(",") if t.strip()]
    for key, val in (("model.trust_remote_code", cfg.model.trust_remote_code),
                     ("run.trust_remote_code", cfg.run.trust_remote_code)):
        if val is not None and not isinstance(val, bool):
            raise ValueError(f"{key} must be true or false, got {val!r}")
    if isinstance(cfg.run.modalities, str):
        cfg.run.modalities = [t.strip() for t in cfg.run.modalities.split(",") if t.strip()]
    for m in cfg.run.modalities or []:
        if m not in ("text", "image", "all"):
            raise ValueError(f"run.modalities entries must be text|image|all, got {m!r}")
    if isinstance(cfg.run.exclude, str):
        cfg.run.exclude = [t.strip() for t in cfg.run.exclude.split(",") if t.strip()]
    if cfg.model.supports_images is False and not cfg.run.modalities:
        cfg.run.modalities = ["text"]
    if isinstance(cfg.run.languages, str):
        cfg.run.languages = [t.strip() for t in cfg.run.languages.split(",") if t.strip()]


def dump_config(cfg: BenchConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
