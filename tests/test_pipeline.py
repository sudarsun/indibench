import json
import os

from indibench.config import load_config
from indibench.runner import Evaluator


def _run(fixture_dir, tmp_path, mode, tasks=("local_all",), extra=()):
    cfg = load_config(None, {
        "model.backend": "dummy", "run.tasks": list(tasks), "run.task_files": [str(fixture_dir / "tasks.yaml")],
        "run.output_dir": str(tmp_path), "run.run_name": f"r_{mode}", "run.bootstrap": 50,
    }, [f"model.extra.mode={mode}", *extra])
    ev = Evaluator(cfg)
    return ev, ev.run()


def test_oracle_scores_perfect(fixture_dir, tmp_path):
    ev, out = _run(fixture_dir, tmp_path, "oracle")
    res = out["results"]
    assert res["local_mcq_ta"]["metrics"]["accuracy"]["value"] == 100.0
    assert res["local_qa_ta"]["metrics"]["exact_match"]["value"] == 100.0
    assert res["local_mt_en-ta"]["metrics"]["chrf"]["value"] > 99
    assert res["local_vqa_xl_ta"]["metrics"]["yesno_accuracy"]["value"] == 100.0
    assert "domain" in res["local_mcq_ta"]["breakdown"]
    for f in ("report.md", "summary.csv", "results.json", "config.yaml"):
        assert os.path.exists(os.path.join(ev.run_dir, f))
    assert len(open(os.path.join(ev.run_dir, "predictions", "local_mcq_ta.jsonl")).readlines()) == 6


def test_limit_resume_and_loglikelihood(fixture_dir, tmp_path):
    ev, out = _run(fixture_dir, tmp_path, "random", ("local_mcq_ta",), ["run.limit=3"])
    assert out["results"]["local_mcq_ta"]["n"] == 3
    # Resume with a larger limit: only the new samples are inferred.
    ev2, out2 = _run(fixture_dir, tmp_path, "random", ("local_mcq_ta",), ["run.limit=6"])
    assert out2["results"]["local_mcq_ta"]["n"] == 6
    ev3, out3 = _run(fixture_dir, tmp_path, "ll", ("local_mcq_ta",), ["run.mcq_mode=loglikelihood"])
    assert out3["results"]["local_mcq_ta"]["mode"] == "loglikelihood"


def test_builtin_synthetic_ocr(tmp_path):
    cfg = load_config(None, {"model.backend": "dummy", "run.tasks": ["synth_ocr_ta"], "run.limit": 4,
                             "run.output_dir": str(tmp_path), "run.run_name": "ocr"}, ["model.extra.mode=oracle"])
    out = Evaluator(cfg).run()
    r = out["results"]["synth_ocr_ta"]
    assert r["metrics"]["cer"]["value"] == 0.0, r


def test_registry_expansion():
    from indibench.tasks.registry import TaskRegistry

    reg = TaskRegistry()
    assert "belebele_ta" in reg.specs and reg.specs["belebele_ta"]["dataset"]["name"] == "tam_Taml"
    assert reg.specs["milu_ta"]["dataset"]["data_dir"] == "Tamil"
    assert reg.specs["cvqa_ta"]["dataset"]["filter"]["Subset"] == "('Tamil', 'India')"
    names = reg.resolve(["indic_text"], ["ta"])
    assert names and all(reg.specs[n].get("lang") in ("ta", None) for n in names)
    assert reg.specs["in22conv_en-ta"]["fields"]["target"] == "tam_Taml|sentence_tam_Taml"


def test_config_precedence(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("model: {backend: dummy}\nrun: {limit: 10, batch_size: 2}\nhardware_profile: l40s\n")
    cfg = load_config(str(y), {"run.limit": 20}, ["generation.max_new_tokens=7"])
    assert cfg.run.limit == 20 and cfg.run.batch_size == 2 and cfg.generation.max_new_tokens == 7
    assert cfg.model.gpu_memory_utilization == 0.90
