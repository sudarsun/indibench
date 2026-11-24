"""Regression tests for (1) IndicXNLI's single-key JSON container, (2) script-based Hub datasets
such as XL-Sum, and (3) text-only task selection."""
import json

import pytest

from indibench.config import ImageConfig, RunConfig, load_config
from indibench.data import loader as L
from indibench.tasks.base import build_task
from indibench.tasks.registry import TaskRegistry


# ------------------------------------------------------------------ IndicXNLI container JSON
def _xnli_file(tmp_path, labels):
    rows = [{"premise": f"முன்கூற்று {i}", "hypothesis": f"கருதுகோள் {i}", "label": lab} for i, lab in enumerate(labels)]
    p = tmp_path / "xnli_ta.json"
    p.write_text(json.dumps({"xnli_ta": rows}, ensure_ascii=False), encoding="utf-8")  # ONE object, one key
    return p


@pytest.mark.parametrize("labels", [[0, 1, 2, 0], ["entailment", "neutral", "contradiction", "entailment"]])
def test_indicxnli_container_json_parses(tmp_path, labels):
    path = _xnli_file(tmp_path, labels)
    spec = TaskRegistry().get("indicxnli_ta")
    spec["dataset"] = {"data_files": str(path), "format": "json", "split": "test"}
    task = build_task(spec, RunConfig(), ImageConfig())
    src = task.load_source()
    assert len(src) == 4 and "premise" in src.column_names
    docs = [task.make_doc(r, i) for i, r in enumerate(src.rows(range(len(src))))]
    assert all(d is not None for d in docs)
    assert [d.gold for d in docs] == [0, 1, 2, 0]


def test_column_oriented_json(tmp_path):
    p = tmp_path / "cols.json"
    p.write_text(json.dumps({"question": ["a", "b", "c"], "answer": ["x", "y", "z"]}))
    src = L.load_rows({"data_files": str(p), "format": "json", "split": "test"})
    assert len(src) == 3 and src.rows([2])[0] == {"question": "c", "answer": "z"}


def test_normal_jsonl_untouched(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps({"q": i, "tags": [{"k": i}]}) for i in range(3)))
    src = L.load_rows({"data_files": str(p), "format": "json", "split": "test"})
    assert len(src) == 3 and src.is_hf          # multi-row files are never "unwrapped"


# ------------------------------------------------------------------ script-based datasets (XL-Sum)
def test_pick_parquet_files_layouts():
    old = ["tamil/xlsum-test.parquet", "tamil/xlsum-train.parquet", "tamil/xlsum-validation.parquet",
           "tamil/train/index.duckdb", "hindi/xlsum-test.parquet"]
    assert L.pick_parquet_files(old, "tamil", "test") == ["tamil/xlsum-test.parquet"]
    assert L.pick_parquet_files(old, "tamil", "validation") == ["tamil/xlsum-validation.parquet"]
    new = ["tamil/test/0000.parquet", "tamil/test/0001.parquet", "tamil/train/0000.parquet"]
    assert L.pick_parquet_files(new, "tamil", "test") == ["tamil/test/0000.parquet", "tamil/test/0001.parquet"]
    both = old + new
    assert L.pick_parquet_files(both, "tamil", "test") == ["tamil/test/0000.parquet", "tamil/test/0001.parquet"]
    assert L.pick_parquet_files(["default/partial-test/0000.parquet"], None, "test") == ["default/partial-test/0000.parquet"]
    assert L.pick_parquet_files(old, "telugu", "test") == []


def test_script_dataset_falls_back_to_converted_parquet(monkeypatch):
    import datasets
    import huggingface_hub

    def fake_raw(dspec, split, streaming, cache_dir, token):
        raise RuntimeError("Dataset scripts are no longer supported, but found xlsum.py")

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo, repo_type, revision):
            assert revision == "refs/convert/parquet" and repo == "csebuetnlp/xlsum"
            return ["tamil/xlsum-test.parquet", "tamil/xlsum-train.parquet", "hindi/xlsum-test.parquet"]

    seen = {}
    real_load = datasets.load_dataset

    def fake_load(builder, data_files=None, split=None, **kw):
        if builder != "parquet":
            return real_load(builder, data_files=data_files, split=split, **kw)
        seen["urls"] = data_files[split]
        return datasets.Dataset.from_list([{"id": "1", "text": "நீண்ட கட்டுரை", "summary": "சுருக்கம்"}])

    monkeypatch.setattr(L, "_load_raw", fake_raw)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(datasets, "load_dataset", fake_load)

    task = build_task(TaskRegistry().get("xlsum_ta"), RunConfig(), ImageConfig())
    src = task.load_source()
    assert seen["urls"] == ["hf://datasets/csebuetnlp/xlsum@refs%2Fconvert%2Fparquet/tamil/xlsum-test.parquet"]
    doc = task.make_doc(src.rows([0])[0], 0)
    assert doc.vars["source"] == "நீண்ட கட்டுரை" and doc.gold == ["சுருக்கம்"]


def test_non_script_errors_are_not_masked(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("401 Client Error: gated dataset")

    monkeypatch.setattr(L, "_load_raw", boom)
    with pytest.raises(L.DatasetLoadError, match="access denied"):
        L.load_rows({"path": "ai4bharat/MILU", "data_dir": "Tamil", "split": "test"})


# ------------------------------------------------------------------ text-only selection
def test_modality_tagging_and_filters():
    reg = TaskRegistry()
    assert reg.specs["belebele_ta"]["modality"] == "text"
    for n in ("synth_ocr_ta", "pope_xl_ta", "mmmu_val", "mathvista_testmini", "chartqa", "cvqa_ta"):
        assert reg.specs[n]["modality"] == "image", n
    text = reg.resolve(["tamil"], ["ta"], modalities=["text"])
    assert text and all(reg.specs[n]["modality"] == "text" for n in text)
    assert "synth_ocr_ta" not in text and "xlsum_ta" in text
    assert reg.resolve(["vision_core"], modalities=["text"]) == []
    assert "xlsum_ta" not in reg.resolve(["tamil_text"], exclude=["xlsum_*"])
    assert reg.resolve(["quick"], exclude=["vision"]) == ["belebele_ta", "belebele_en", "indicsentiment_ta"]


def test_text_only_run_skips_vision(fixture_dir, tmp_path):
    from indibench.runner import Evaluator

    base = {"model.backend": "dummy", "run.task_files": [str(fixture_dir / "tasks.yaml")],
            "run.tasks": ["local_all"], "run.output_dir": str(tmp_path)}
    # 1) explicit --text-only
    cfg = load_config(None, dict(base, **{"run.modalities": ["text"], "run.run_name": "t1"}), [])
    out = Evaluator(cfg).run()["results"]
    assert "local_vqa_xl_ta" not in out and "local_mcq_ta" in out
    # 2) model declared text-only -> vision tasks dropped automatically
    cfg = load_config(None, dict(base, **{"run.run_name": "t2"}), ["model.supports_images=false"])
    assert cfg.run.modalities == ["text"]
    out = Evaluator(cfg).run()["results"]
    assert "local_vqa_xl_ta" not in out and "local_qa_ta" in out
