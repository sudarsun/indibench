"""GPU-utilisation features: batched log-likelihood, auto batch size, prefetch pipeline, monitor hints."""
import pytest

from indibench.gpumon import summarise, utilisation_hints


def test_summarise_and_hints():
    samples = [[(5.0, 4.0, 48.0)]] * 10 + [[(40.0, 6.0, 48.0)]] * 10
    st = summarise(samples)
    assert st["util_avg"] == 22.5 and st["idle_share"] == 50.0 and st["mem_peak_pct"] == 12.5
    hints = utilisation_hints(st, "hf", 8, "image")
    assert any("batch-size" in h for h in hints) and any("max-pixels" in h for h in hints)
    assert utilisation_hints(dict(st, util_avg=90), "hf", 8, "text") == []
    assert any("concurrency" in h for h in utilisation_hints(st, "openai", 8, "text"))


def test_batch_size_auto_and_validation(fixture_dir, tmp_path):
    from indibench.config import load_config
    from indibench.runner import Evaluator

    cfg = load_config(None, {"model.backend": "dummy", "run.tasks": ["local_mcq_ta"],
                             "run.task_files": [str(fixture_dir / "tasks.yaml")], "run.limit": 5,
                             "run.batch_size": "auto", "run.output_dir": str(tmp_path), "run.prefetch": 1}, [])
    res = Evaluator(cfg).run()["results"]["local_mcq_ta"]
    assert res["n"] == 5 and res["batch_size"] == 64
    with pytest.raises(ValueError):
        load_config(None, {"run.batch_size": 0}, [])
    with pytest.raises(ValueError):
        load_config(None, {"run.batch_size": "big"}, [])


def test_prefetch_matches_across_settings(fixture_dir, tmp_path):
    """Results must not depend on prefetch depth / chunking."""
    from indibench.config import load_config
    from indibench.runner import Evaluator

    outs = []
    for k, (pf, ch, bs) in enumerate([(1, 1, 1), (4, 7, 3), (2, None, 16)]):
        cfg = load_config(None, {"model.backend": "dummy", "model.extra": {"mode": "oracle"},
                                 "run.tasks": ["local_mcq_ta", "local_vqa_xl_ta"],
                                 "run.task_files": [str(fixture_dir / "tasks.yaml")], "run.limit": 6,
                                 "run.batch_size": bs, "run.prefetch": pf, "run.inference_chunk": ch,
                                 "run.output_dir": str(tmp_path), "run.run_name": f"r{k}"}, [])
        r = Evaluator(cfg).run()["results"]
        outs.append({t: (v["n"], v["metrics"]) for t, v in r.items()})
    assert outs[0] == outs[1] == outs[2]


def test_hf_batched_loglikelihood_equals_unbatched(tiny_llava):
    torch = pytest.importorskip("torch")
    from indibench.config import ModelConfig
    from indibench.models.base import Request
    from indibench.models.hf import HFModel

    m = HFModel(ModelConfig(backend="hf", name_or_path=tiny_llava, dtype="float32", device_map="cpu"))
    reqs = [Request(messages=[{"role": "user", "content": [{"type": "text", "text": t}]}], continuations=[" A", " B", " yes no"])
            for t in ["Answer A B", "இன்று வானிலை சென்னை தமிழ்நாட்டின் Answer with the option's letter", "C"]]
    m._cap = 1
    single = m.loglikelihood(reqs)
    m.reset_capacity()
    batched = m.loglikelihood(reqs)
    for a, b in zip(single, batched):
        for (la, ga), (lb, gb) in zip(a, b):
            assert la == pytest.approx(lb, abs=1e-3) and ga == gb


def test_report_gpu_section():
    from indibench.report import render_markdown

    res = {"t": {"n": 10, "metrics": {}, "primary_metric": None, "batch_size": 8,
                 "timing": {"samples_per_s": 2.5},
                 "gpu": {"util_avg": 30.0, "idle_share": 40.0, "mem_peak_pct": 20.0, "hints": ["raise x"]}}}
    md = render_markdown({"results": res, "config": {}, "meta": {}})
    assert "## GPU utilisation" in md and "raise x" in md
