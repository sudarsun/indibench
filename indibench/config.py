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
    attn_implementation: Optional[str] = "sdpa"  # sdpa | flash_attention_2 | eager
    quantization: Optional[str] = None           # 4bit | 8bit (bitsandbytes, hf) | awq/gptq/fp8 (vllm)
    trust_remote_code: bool = False
    chat_template: bool = True                   # apply the model's chat template
    system_prompt: Optional[str] = None
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
    hardware_profile: Optional[str] = None             # l40s | a100-40g | a100-80g | auto

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
}


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
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0).lower()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return None
    if "l40" in name:
        return "l40s"
    if "a100" in name or "h100" in name or "h200" in name:
        return "a100-80g" if mem_gb > 60 else "a100-40g"
    if mem_gb >= 70:
        return "a100-80g"
    if mem_gb >= 44:
        return "l40s"
    return "a100-40g"


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
    profile = user.get("hardware_profile")
    if profile == "auto":
        profile = detect_hardware_profile()
        user["hardware_profile"] = profile
    if profile:
        if profile not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware_profile {profile!r}; choose from {sorted(HARDWARE_PROFILES)} or 'auto'")
        for k, v in HARDWARE_PROFILES[profile].items():
            if _get_path(user, k) is None:
                _set_path(user, k, v)

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
    if isinstance(cfg.run.languages, str):
        cfg.run.languages = [t.strip() for t in cfg.run.languages.split(",") if t.strip()]


def dump_config(cfg: BenchConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
