"""Adaptive max_new_tokens, --help defaults, per-task timing, memory release between tasks."""
import pytest

from indibench.config import load_config
from indibench.runner import Evaluator, _is_looping


def _cfg(fixture_dir, tmp_path, **kv):
    base = {"model.backend": "dummy", "run.task_files": [str(fixture_dir / "tasks.yaml")], "run.limit": 4,
            "run.output_dir": str(tmp_path), "run.tasks": ["local_vqa_xl_ta", "local_mcq_ta"]}
    base.update(kv)
    return load_config(None, base, [])


def test_truncated_answers_are_extended(fixture_dir, tmp_path):
    cfg = _cfg(fixture_dir, tmp_path, **{"model.extra": {"needs_tokens": 1500}, "run.run_name": "ext"})
    res = Evaluator(cfg).run()["results"]
    for name in ("local_vqa_xl_ta", "local_mcq_ta"):
        st = res[name]["max_new_tokens"]
        assert st["truncated"] == 0 and st["extended"] > 0 and st["max_used"] >= 1500


def test_extension_respects_limit_and_disable(fixture_dir, tmp_path):
    cfg = _cfg(fixture_dir, tmp_path, **{"model.extra": {"needs_tokens": 5000}, "run.run_name": "lim",
                                         "generation.max_new_tokens_limit": 1024})
    st = Evaluator(cfg).run()["results"]["local_mcq_ta"]["max_new_tokens"]
    assert st["max_used"] == 1024 and st["truncated"] == 4
    cfg = _cfg(fixture_dir, tmp_path, **{"model.extra": {"needs_tokens": 5000}, "run.run_name": "off",
                                         "generation.extend_on_truncation": False})
    st = Evaluator(cfg).run()["results"]["local_mcq_ta"]["max_new_tokens"]
    assert st["extended"] == 0 and st["truncated"] == 4


def test_budget_floor_and_reference_scaling(fixture_dir, tmp_path):
    from indibench.models.base import BaseModel
    from indibench.tasks.base import build_task

    cfg = _cfg(fixture_dir, tmp_path, **{"generation.max_new_tokens": 700})
    ev = Evaluator(cfg)
    task = build_task(ev.registry.get("local_mcq_ta"), cfg.run, cfg.model.image)
    assert task.token_budget(cfg.generation) == 700            # integer = floor for every task
    cfg.generation.max_new_tokens = "auto"
    assert task.token_budget(cfg.generation) == task.max_new_tokens or task.default_max_new_tokens
    # heuristic counter: Tamil text costs ~1 token per character
    assert BaseModel(cfg.model).count_tokens(["சென்னை"])[0] >= 6
    with pytest.raises(ValueError):
        load_config(None, {"model.backend": "dummy", "generation.max_new_tokens": "lots"}, [])


def test_loop_detection():
    assert _is_looping("The answer is " + "yes yes " * 60)
    assert not _is_looping("Step 1: add the numbers. " + " ".join(f"Then {i}+{i}={2*i}." for i in range(40)))


def test_timing_and_release(fixture_dir, tmp_path):
    cfg = _cfg(fixture_dir, tmp_path, **{"run.run_name": "time"})
    ev = Evaluator(cfg)
    from indibench.models.base import load_model

    ev.model = load_model(cfg.model)
    calls = []
    ev.model.release_memory = lambda: calls.append(1)
    payload = ev.run()
    assert len(calls) == 2                                      # once after every task
    for r in payload["results"].values():
        t = r["timing"]
        assert t["seconds"] >= t["inference_s"] >= 0 and "load_s" in t
    md = open(f"{ev.run_dir}/report.md", encoding="utf-8").read()
    assert "| **total** |" in md and "total time:" in md


def test_help_shows_defaults(capsys):
    from indibench.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--help"])
    out = capsys.readouterr().out
    assert "[default: auto]" in out                              # --max-new-tokens
    assert "[default: 8192]" in out                              # --max-new-tokens-limit
    assert "[default: 0.9]" in out                               # --gpu-mem-util
    assert "default: None" not in out


def test_hf_reports_truncation_and_counts_tokens(tiny_llava):
    from indibench.config import GenerationConfig, ModelConfig
    from indibench.models.base import Request
    from indibench.models.hf import HFModel

    m = HFModel(ModelConfig(backend="hf", name_or_path=tiny_llava, dtype="float32", device_map="cpu"))
    reqs = [Request(messages=[{"role": "user", "content": [{"type": "text", "text": t}]}], max_new_tokens=3)
            for t in ("Answer A B", "இன்று வானிலை")]
    out = m.generate_ex(reqs, GenerationConfig())
    assert len(out) == 2 and all(isinstance(t, str) and isinstance(f, bool) for t, f in out)
    assert any(f for _, f in out)            # a random tiny model does not stop within 3 tokens
    assert all(n > 0 for n in m.count_tokens(["சென்னை", "Chennai"]))
    assert m.context_length() == 1024
    m.release_memory()
