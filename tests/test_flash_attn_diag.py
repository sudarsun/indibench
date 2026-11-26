"""FlashAttention-2 diagnosis and generate() kwargs."""
import importlib.metadata as md
import importlib.util
import types

import pytest

torch = pytest.importorskip("torch")
from indibench.hardware import flash_attn_status  # noqa: E402


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (8, 9))


def test_old_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 5))
    st = flash_attn_status()
    assert not st["ok"] and "sm_75" in st["reason"]


def test_not_installed(fake_gpu, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "flash_attn" else real(n, *a))
    st = flash_attn_status()
    assert not st["ok"] and "not importable" in st["reason"] and "--no-build-isolation" in st["fix"]


def test_missing_metadata(fake_gpu, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: types.SimpleNamespace() if n == "flash_attn" else real(n, *a))

    def version(name):
        raise md.PackageNotFoundError(name)
    monkeypatch.setattr(md, "version", version)
    st = flash_attn_status()
    assert not st["ok"] and "metadata" in st["reason"]


def test_editable_install_without_module_record(fake_gpu, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: types.SimpleNamespace() if n == "flash_attn" else real(n, *a))
    monkeypatch.setattr(md, "version", lambda name: "2.8.4")
    monkeypatch.setattr(md, "packages_distributions", lambda: {})
    st = flash_attn_status()
    assert not st["ok"] and "editable" in st["reason"] and st["version"] == "2.8.4"


def test_abi_mismatch(fake_gpu, monkeypatch):
    import importlib

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: types.SimpleNamespace() if n == "flash_attn" else real(n, *a))
    monkeypatch.setattr(md, "version", lambda name: "2.8.4")
    monkeypatch.setattr(md, "packages_distributions", lambda: {"flash_attn": ["flash_attn"]})
    real_import = importlib.import_module

    def imp(name, *a):
        if name == "flash_attn_2_cuda":
            raise ImportError("flash_attn_2_cuda.so: undefined symbol: _ZN3c104cuda9SetDevice")
        return real_import(name, *a)
    monkeypatch.setattr(importlib, "import_module", imp)
    st = flash_attn_status()
    assert not st["ok"] and "undefined symbol" in st["reason"] and "different PyTorch" in st["reason"]


def test_generate_passes_max_length_none(tiny_llava, monkeypatch):
    from indibench.config import GenerationConfig, ModelConfig
    from indibench.models.base import Request
    from indibench.models.hf import HFModel

    m = HFModel(ModelConfig(backend="hf", name_or_path=tiny_llava, dtype="float32", device_map="cpu"))
    seen = {}
    real = m.model.generate

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(m.model, "generate", spy)
    m.generate([Request(messages=[{"role": "user", "content": [{"type": "text", "text": "A B"}]}],
                        max_new_tokens=2)], GenerationConfig())
    assert "max_length" in seen and seen["max_length"] is None and seen["max_new_tokens"] == 2
