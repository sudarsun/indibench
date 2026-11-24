import json

import pytest

from indibench.config import ImageConfig, RunConfig, load_config
from indibench.data import loader as L
from indibench.tasks.base import _get, build_task
from indibench.tasks.registry import TaskRegistry


# ------------------------------------------------------------------ trust_remote_code config
def _cfg(cli=None, sets=()):
    base = {"model.backend": "hf", "model.name_or_path": "x", "hardware_profile": "none"}
    base.update(cli or {})
    return load_config(None, base, list(sets))


def test_trust_remote_code_defaults_off():
    cfg = _cfg()
    assert cfg.model.trust_remote_code is False and cfg.run.trust_remote_code is None


def test_top_level_shorthand_sets_both_and_specific_keys_win():
    cfg = _cfg(sets=["trust_remote_code=true"])
    assert cfg.model.trust_remote_code is True and cfg.run.trust_remote_code is True
    cfg = _cfg({"trust_remote_code": True, "run.trust_remote_code": False})
    assert cfg.model.trust_remote_code is True and cfg.run.trust_remote_code is False
    cfg = _cfg({"trust_remote_code": False})
    assert cfg.model.trust_remote_code is False and cfg.run.trust_remote_code is False


def test_trust_remote_code_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("model: {backend: hf, name_or_path: x, trust_remote_code: true}\nrun: {trust_remote_code: false}\n")
    cfg = load_config(str(y), {"hardware_profile": "none"}, [])
    assert cfg.model.trust_remote_code is True and cfg.run.trust_remote_code is False


def test_trust_remote_code_rejects_non_bool():
    with pytest.raises(ValueError):
        _cfg(sets=["model.trust_remote_code=maybe"])


def test_cli_flag_parsing():
    from indibench.cli import build_parser

    p = build_parser()
    a = p.parse_args(["run", "--trust-remote-code"])
    assert a.trust_remote_code is True
    a = p.parse_args(["run", "--trust-remote-code", "false", "--dataset-trust-remote-code", "true"])
    assert a.trust_remote_code is False and getattr(a, "run.trust_remote_code") is True
    a = p.parse_args(["run", "--model-trust-remote-code", "true"])
    assert getattr(a, "model.trust_remote_code") is True


def test_dataset_flag_reaches_load_dataset(monkeypatch):
    import datasets

    seen = {}

    def fake_load(path, split=None, **kw):
        seen.update(kw)
        return datasets.Dataset.from_list([{"a": 1}])

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(L, "datasets_supports_remote_code", lambda: True)        # simulate datasets<4
    L._load_raw({"path": "org/scripted", "trust_remote_code": True}, "test", False, None, None)
    assert seen["trust_remote_code"] is True
    seen.clear()
    monkeypatch.setattr(L, "datasets_supports_remote_code", lambda: False)       # datasets>=4
    L._load_raw({"path": "org/scripted", "trust_remote_code": True}, "test", False, None, None)
    assert "trust_remote_code" not in seen                                       # never passed -> no TypeError


def test_run_default_propagates_to_task_dataset_spec():
    reg = TaskRegistry()
    t = build_task(reg.get("xlsum_ta"), RunConfig(trust_remote_code=True), ImageConfig())
    assert t.dataset_spec()["trust_remote_code"] is True
    spec = reg.get("xlsum_ta")
    spec["dataset"]["trust_remote_code"] = False                                  # per-task value wins
    assert build_task(spec, RunConfig(trust_remote_code=True), ImageConfig()).dataset_spec()["trust_remote_code"] is False
    assert "trust_remote_code" not in build_task(reg.get("xlsum_ta"), RunConfig(), ImageConfig()).dataset_spec()


# ------------------------------------------------------------------ IN22-Conv columns
def test_field_alternatives():
    assert _get({"eng_Latn": "hi"}, "eng_Latn|sentence_eng_Latn") == "hi"
    assert _get({"sentence_eng_Latn": "hi"}, "eng_Latn|sentence_eng_Latn") == "hi"
    assert _get({"x": 1}, "eng_Latn|sentence_eng_Latn") is None


@pytest.mark.parametrize("prefix", ["", "sentence_"])
def test_in22conv_accepts_both_column_styles(tmp_path, prefix):
    rows = [{"id": i, "domain": "culture", f"{prefix}eng_Latn": f"Hello {i}", f"{prefix}tam_Taml": f"வணக்கம் {i}"}
            for i in range(3)]
    p = tmp_path / "in22.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    spec = TaskRegistry().get("in22conv_en-ta")
    assert spec["fields"]["source"] == "eng_Latn|sentence_eng_Latn"
    assert spec["fields"]["target"] == "tam_Taml|sentence_tam_Taml"
    spec["dataset"] = {"path": str(p), "split": "test"}
    task = build_task(spec, RunConfig(), ImageConfig())
    src = task.load_source()
    docs = [task.make_doc(r, i) for i, r in enumerate(src.rows(range(len(src))))]
    assert all(docs) and docs[2].vars["source"] == "Hello 2" and docs[2].gold == ["வணக்கம் 2"]
