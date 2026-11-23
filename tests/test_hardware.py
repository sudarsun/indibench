from indibench.config import load_config
from indibench.hardware import GPUInfo, check_fit, guess_params_b, harden_config, profile_for

RTX2080TI = GPUInfo("NVIDIA GeForce RTX 2080 Ti", (7, 5), 10.75)
RTX3090 = GPUInfo("NVIDIA GeForce RTX 3090", (8, 6), 24.0)
RTX4090 = GPUInfo("NVIDIA GeForce RTX 4090", (8, 9), 23.99)
L40S = GPUInfo("NVIDIA L40S", (8, 9), 44.5)
A100_40 = GPUInfo("NVIDIA A100-SXM4-40GB", (8, 0), 39.4)
T4 = GPUInfo("Tesla T4", (7, 5), 15.0)
A100_80 = GPUInfo("NVIDIA A100-SXM4-80GB", (8, 0), 79.2)
H100 = GPUInfo("NVIDIA H100 80GB HBM3", (9, 0), 79.1)
H100_NVL = GPUInfo("NVIDIA H100 NVL", (9, 0), 93.6)
H200 = GPUInfo("NVIDIA H200", (9, 0), 139.8)
H100_MIG = GPUInfo("NVIDIA H100 80GB HBM3 MIG 1g.10gb", (9, 0), 9.8)


def test_capabilities():
    assert not RTX2080TI.bf16 and not RTX2080TI.flash_attn2 and not RTX2080TI.fp8
    assert RTX3090.bf16 and RTX3090.flash_attn2 and not RTX3090.fp8
    assert RTX4090.bf16 and RTX4090.fp8 and RTX4090.arch == "Ada"
    assert RTX2080TI.arch == "Turing" and RTX2080TI.supported_by_torch_cu13
    assert not GPUInfo("GTX 1080 Ti", (6, 1), 11).supported_by_torch_cu13


def test_profile_mapping():
    assert profile_for(RTX2080TI) == "rtx2080ti"
    assert profile_for(RTX3090) == "rtx3090"
    assert profile_for(RTX4090) == "rtx4090"
    assert profile_for(L40S) == "l40s"
    assert profile_for(A100_40) == "a100-40g"
    assert profile_for(T4) == "rtx2080ti"                     # unknown Turing card -> fp16 profile
    assert profile_for(GPUInfo("RTX A5000", (8, 6), 24)) == "rtx3090"
    assert profile_for(A100_80) == "a100-80g"
    assert profile_for(H100) == "h100" and profile_for(H100_NVL) == "h100"
    assert profile_for(H200) == "h200"
    assert profile_for(GPUInfo("NVIDIA GH200 480GB", (9, 0), 95.0)) == "h200"
    assert profile_for(H100_MIG) == "rtx2080ti-bf16"                 # MIG slice: small but bf16
    assert profile_for(GPUInfo("NVIDIA B200", (10, 0), 179.0)) == "h200"   # unknown big card by memory


def test_datacenter_capabilities_and_fit():
    for g in (L40S, A100_40, A100_80, H100, H200):
        assert g.bf16 and g.flash_attn2
    assert H100.fp8 and H200.fp8 and L40S.fp8 and not A100_80.fp8
    assert H100.arch == "Hopper"
    assert check_fit("Qwen2.5-VL-72B-Instruct", "bfloat16", None, H200).verdict == "no"
    assert check_fit("Qwen2.5-VL-72B-Instruct", "bfloat16", "fp8", H200).verdict == "ok"
    assert check_fit("Qwen2.5-VL-72B-Instruct", "bfloat16", "4bit", H100).verdict == "ok"
    assert check_fit("Qwen2.5-VL-32B-Instruct", "bfloat16", None, H200).verdict == "ok"
    cfg = load_config(None, {"model.backend": "vllm", "model.name_or_path": "x", "hardware_profile": "h200"}, [])
    assert cfg.run.batch_size == 24 and cfg.model.max_model_len == 32768
    cfg = load_config(None, {"model.backend": "vllm", "model.name_or_path": "x", "hardware_profile": "H100-NVL"}, [])
    assert cfg.hardware_profile == "h100"


def test_param_guess():
    assert guess_params_b("Qwen/Qwen2.5-VL-7B-Instruct") == 7
    assert guess_params_b("google/gemma-3-12b-it") == 12
    assert guess_params_b("Qwen/Qwen3-VL-30B-A3B-Instruct") == 30
    assert guess_params_b("HuggingFaceTB/SmolVLM-Instruct") is None


def test_fit_advice():
    assert check_fit("Qwen2.5-VL-7B", "float16", None, RTX2080TI).verdict == "no"
    assert "4bit" in check_fit("Qwen2.5-VL-7B", "float16", None, RTX2080TI).advice
    assert check_fit("Qwen2.5-VL-7B", "float16", "4bit", RTX2080TI).verdict == "ok"
    assert check_fit("Qwen2.5-VL-3B", "float16", None, RTX2080TI).verdict == "ok"
    assert check_fit("Qwen2.5-VL-7B", "bfloat16", None, RTX3090).verdict == "ok"
    assert "8bit" in check_fit("gemma-3-12b-it", "bfloat16", None, RTX4090).advice
    assert check_fit("Qwen2.5-VL-32B", "bfloat16", "4bit", RTX3090).verdict == "tight"


def _cfg(**sets):
    over = [f"{k}={v}" for k, v in sets.items()]
    return load_config(None, {"model.backend": "hf", "model.name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
                              "hardware_profile": "none"}, over)


def test_harden_turing():
    cfg = _cfg(**{"model.attn_implementation": "flash_attention_2"})
    notes = harden_config(cfg, RTX2080TI)
    assert cfg.model.dtype == "float16" and cfg.model.attn_implementation == "sdpa"
    assert any("bfloat16" in n for n in notes)


def test_harden_gemma_fp16_warning_and_fp8():
    cfg = load_config(None, {"model.backend": "vllm", "model.name_or_path": "google/gemma-3-4b-it",
                             "hardware_profile": "none", "model.quantization": "fp8"}, [])
    notes = harden_config(cfg, RTX2080TI)
    assert any("overflow" in n for n in notes) and any("FP8" in n for n in notes)
    assert any("Turing" in n for n in notes)


def test_harden_ampere_untouched():
    cfg = _cfg(**{"model.attn_implementation": "flash_attention_2"})
    assert harden_config(cfg, RTX3090) == []
    assert cfg.model.dtype == "bfloat16" and cfg.model.attn_implementation == "flash_attention_2"


def test_profiles_and_aliases():
    cfg = load_config(None, {"model.backend": "hf", "model.name_or_path": "x", "hardware_profile": "2080ti"}, [])
    assert cfg.hardware_profile == "rtx2080ti" and cfg.model.dtype == "float16" and cfg.run.batch_size == 2
    cfg = load_config(None, {"model.backend": "hf", "model.name_or_path": "x", "hardware_profile": "4090",
                             "run.batch_size": 9}, [])
    assert cfg.run.batch_size == 9 and cfg.model.max_model_len == 8192   # explicit value wins
    cfg = load_config(None, {"model.backend": "dummy"}, [])
    assert cfg.hardware_profile is None                                # auto skipped for non-GPU backends
