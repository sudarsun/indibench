"""Exercises the transformers backend with a tiny randomly-initialised LLaVA built locally (no downloads).
Skipped when torch/transformers are not installed."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_hf_backend_generate_and_loglikelihood(tiny_llava, fixture_dir, tmp_path):
    from indibench.config import load_config
    from indibench.runner import Evaluator

    for mode in ("generate", "loglikelihood"):
        cfg = load_config(None, {
            "model.backend": "hf", "model.name_or_path": tiny_llava, "model.dtype": "float32",
            "model.device_map": "cpu", "run.tasks": ["local_mcq_ta", "local_vqa_xl_ta"],
            "run.task_files": [str(fixture_dir / "tasks.yaml")], "run.limit": 3, "run.batch_size": 2,
            "run.output_dir": str(tmp_path), "run.run_name": mode, "run.mcq_mode": mode,
            "generation.max_new_tokens": 4, "generation.max_new_tokens_limit": 8}, [])
        res = Evaluator(cfg).run()["results"]
        assert res["local_mcq_ta"]["n"] == 3 and res["local_mcq_ta"]["mode"] == mode
        assert res["local_vqa_xl_ta"]["n"] == 3
