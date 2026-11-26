"""Executive summary, IFEval checkers, and the loader/spec fixes from 0.3."""
import json

import pytest

from indibench.config import ImageConfig, RunConfig
from indibench.data import loader as L
from indibench.summary import analyse, area_of, band, render_html, skill_score, write_executive_summary
from indibench.tasks.base import build_task
from indibench.tasks.ifeval import check_instructions
from indibench.tasks.registry import TaskRegistry


# ------------------------------------------------------------------ executive summary
def _res(fam, lang, typ, tags, value, metric="accuracy", chance=None, modality="text", **kw):
    r = {"family": fam, "lang": lang, "type": typ, "tags": tags, "modality": modality, "n": 100,
         "primary_metric": metric, "metrics": {metric: {"value": value, "ci_low": value - 5, "ci_high": value + 5,
                                                         "higher_is_better": metric != "cer"}}}
    if chance is not None:
        r["chance"] = chance
    r.update(kw)
    return r


def _payload(shift=0.0):
    return {"model": {"name": "m"}, "env": {}, "config": {"run": {"limit": 100}}, "results": {
        "belebele_ta": _res("belebele", "ta", "mcq", ["indic"], 70 + shift, chance=25),
        "belebele_en": _res("belebele", "en", "mcq", ["indic"], 85 + shift, chance=25),
        "flores_en-ta": _res("flores_en2xx", "ta", "generation", ["indic", "translation"], 40 + shift, "chrf",
                             metrics_extra=None),
        "synth_ocr_ta": _res("synth_ocr", "ta", "synthetic_ocr", ["vision"], 20, "cer", modality="image"),
        "gsm8k": _res("gsm8k", None, "math", ["english", "math"], 80 + shift),
        "humaneval": _res("humaneval", None, "code", ["code"], 60 + shift, "pass@1"),
        "ifeval": _res("ifeval", None, "ifeval", ["english", "instruction_following"], 70, "prompt_strict"),
        "medmcqa": _res("medmcqa", None, "mcq", ["indian_english", "medical"], 55, chance=25),
        "broken": {"error": "boom"},
    }}


def test_skill_score_chance_correction_and_inversion():
    assert skill_score(_res("x", None, "mcq", [], 25, chance=25)) == 0.0
    assert skill_score(_res("x", None, "mcq", [], 100, chance=25)) == 100.0
    assert skill_score(_res("x", None, "mcq", [], 62.5, chance=25)) == 50.0
    assert skill_score(_res("x", None, "ocr", [], 20, "cer")) == 80.0
    assert skill_score(_res("x", None, "generation", [], 40, "chrf")) == 40.0


def test_area_routing():
    p = _payload()["results"]
    assert area_of(p["belebele_ta"]) == "tamil" and area_of(p["flores_en-ta"]) == "tamil_gen"
    assert area_of(p["synth_ocr_ta"]) == "vision" and area_of(p["humaneval"]) == "code"
    assert area_of(p["ifeval"]) == "safety" and area_of(p["medmcqa"]) == "stem"
    assert band(75)[0] == "Strong" and band(10)[0] == "Weak" and band(None)[0] == "Not tested"


def test_analyse_parity_baseline_and_render(tmp_path):
    a = analyse(_payload(), baseline=_payload(-10))
    assert a["parity"] == pytest.approx(100 * (60 / 80), rel=1e-6)       # chance-corrected ta / en
    assert a["areas"]["code"]["delta"] == pytest.approx(10)
    assert a["areas"]["indic"]["score"] is None and a["errors"] == ["broken"]
    html = render_html(_payload(), a)
    assert "Overall score" in html and "Δ" in html and "Not evaluated" in html
    path = write_executive_summary(_payload(), str(tmp_path))
    assert path.endswith(".html") and "Tamil understanding" in open(path, encoding="utf-8").read()
    md = write_executive_summary(_payload(), str(tmp_path), fmt="md")
    assert "| Capability |" in open(md, encoding="utf-8").read()


def test_summary_cli(tmp_path):
    from indibench.cli import main

    (tmp_path / "results.json").write_text(json.dumps(_payload()))
    with pytest.raises(SystemExit) as e:
        main(["summary", str(tmp_path)])
    assert e.value.code == 0 and (tmp_path / "executive_summary.html").exists()


# ------------------------------------------------------------------ IFEval
@pytest.mark.parametrize("iid,kw,good,bad", [
    ("punctuation:no_comma", {}, "No commas here.", "Yes, commas."),
    ("length_constraints:number_words", {"relation": "at least", "num_words": 5}, "one two three four five", "too short"),
    ("length_constraints:number_words", {"relation": "less than", "num_words": 3}, "two words", "this has four words"),
    ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', "not json"),
    ("detectable_format:title", {}, "<<My Title>>\nbody", "no title"),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n* b", "* a"),
    ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*one* and *two*", "*one*"),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] at [address]", "[name]"),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Hi\nP.S. bye", "Hi"),
    ("keywords:existence", {"keywords": ["chennai", "idli"]}, "Chennai has idli", "Chennai only"),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "all good", "this is bad"),
    ("keywords:frequency", {"keyword": "tea", "frequency": 2, "relation": "at least"}, "tea tea", "tea"),
    ("keywords:letter_frequency", {"letter": "z", "let_frequency": 2, "let_relation": "at least"}, "zoo zap", "zoo"),
    ("change_case:english_lowercase", {}, "all lower here", "Not Lower"),
    ("change_case:english_capital", {}, "ALL CAPS", "Not caps"),
    ("startend:end_checker", {"end_phrase": "Any other questions?"}, "Done. Any other questions?", "Done."),
    ("startend:quotation", {}, '"quoted"', "plain"),
    ("combination:two_responses", {}, "first\n******\nsecond", "only one"),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Write a poem"}, "Write a poem\nRoses...", "Roses..."),
    ("detectable_format:constrained_response", {}, "My answer is yes.", "Sure"),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "a\n***\nb", "a\n***\nb\n***\nc"),
    ("language:response_language", {"language": "ta"}, "இது தமிழ் பதில்", "This is English"),
    ("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 2},
     "Section 1\na\nSection 2\nb", "Section 1\na"),
])
def test_ifeval_checkers(iid, kw, good, bad):
    assert check_instructions(good, [iid], [kw])[0] == [True]
    assert check_instructions(bad, [iid], [kw])[0] == [False]


def test_ifeval_loose_and_unknown():
    # loose mode tolerates a chatty first line
    s, l_ = check_instructions("Sure! Here it is:\n<<Title>>", ["startend:quotation"], [{}])
    assert s == [False] and l_ == [False]
    s, l_ = check_instructions("Sure, here you go\nall lowercase text", ["change_case:english_lowercase"], [{}])
    assert s == [False] and l_ == [True]
    s, _ = check_instructions("x", ["made_up:instruction"], [{}])
    assert s == [None]


def test_ifeval_task_aggregation(tmp_path):
    rows = [{"key": 1, "prompt": "Answer without commas.", "instruction_id_list": ["punctuation:no_comma"],
             "kwargs": [{"num_words": None}]},
            {"key": 2, "prompt": "Two rules.", "instruction_id_list": ["punctuation:no_comma", "startend:quotation"],
             "kwargs": [{}, {}]}]
    p = tmp_path / "if.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    spec = TaskRegistry().get("ifeval")
    spec["dataset"] = {"path": str(p), "split": "test"}
    t = build_task(spec, RunConfig(bootstrap=0), ImageConfig())
    docs = [t.make_doc(r, i) for i, r in enumerate(t.load_source().rows([0, 1]))]
    recs = []
    for d, resp in zip(docs, ["no commas", "\"quoted, but comma\""]):
        pred, m = t.score(d, resp)
        recs.append({"metrics": m})
    agg = t.aggregate(recs, 0)
    assert agg["prompt_strict"]["value"] == 50.0 and agg["inst_strict"]["value"] == pytest.approx(200 / 3)


# ------------------------------------------------------------------ loader / spec fixes
def test_filter_alternative_columns_and_missing_column(tmp_path):
    p = tmp_path / "multi.jsonl"
    p.write_text("\n".join(json.dumps({"lang": l, "question": "q", "passage": "p", "answer": a})
                           for l, a in [("ta", True), ("hi", False), ("Tamil", False)]))
    src = L.load_rows({"path": str(p), "split": "test", "filter": {"language|lang": ["ta", "Tamil"]}})
    assert len(src) == 2
    with pytest.raises(L.DatasetLoadError, match="Dataset columns"):
        L.load_rows({"path": str(p), "split": "test", "filter": {"language": "ta"}})


def test_boolq_single_table(tmp_path):
    p = tmp_path / "boolq.jsonl"
    p.write_text("\n".join(json.dumps({"language": l, "question": "q?", "passage": "p", "answer": a})
                           for l, a in [("ta", True), ("hi", False), ("ta", False)]))
    spec = TaskRegistry().get("boolq_indic_ta")
    spec["dataset"] = dict(spec["dataset"], path=str(p))
    t = build_task(spec, RunConfig(), ImageConfig())
    src = t.load_source()
    docs = [t.make_doc(r, i) for i, r in enumerate(src.rows(range(len(src))))]
    assert [d.gold for d in docs] == [0, 1]


def test_config_fallback_filter(monkeypatch):
    import datasets

    def fake_raw(dspec, split, streaming, cache_dir, token):
        if dspec.get("name"):
            raise ValueError("BuilderConfig 'ta' not found. Available: ['default']")
        return datasets.Dataset.from_list([{"language": "ta", "question": "q", "choices": ["a", "b"], "answer": 1},
                                           {"language": "hi", "question": "q", "choices": ["a", "b"], "answer": 0}])

    monkeypatch.setattr(L, "_load_raw", fake_raw)
    src = L.load_rows({"path": "sarvamai/mmlu-indic", "name": "ta", "split": "test",
                       "config_fallback_filter": {"language|lang": ["ta"]}})
    assert len(src) == 1 and src.rows([0])[0]["language"] == "ta"


def test_hub_files_pattern(monkeypatch):
    import huggingface_hub

    class Api:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo, repo_type, revision=None):
            return ["README.md", "crosssum_english-ta_test.json", "crosssum_english-ta_train.json",
                    "crosssum_english-hi_test.json"]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    urls = L.resolve_hub_files({"repo": "google/IndicGenBench_crosssum_in", "pattern": "*english-ta_test.json"})
    assert urls == ["hf://datasets/google/IndicGenBench_crosssum_in/crosssum_english-ta_test.json"]
    with pytest.raises(L.DatasetLoadError, match="Similar files"):
        L.resolve_hub_files({"repo": "google/IndicGenBench_crosssum_in", "pattern": "*english-xx_test.json"})


def test_crosssum_examples_file(tmp_path):
    p = tmp_path / "crosssum_english-ta_test.json"
    p.write_text(json.dumps({"canary": "x", "examples": [{"text": "Long English article.", "summary": "தமிழ் சுருக்கம்",
                                                          "lang": "ta"}]}, ensure_ascii=False), encoding="utf-8")
    spec = TaskRegistry().get("crosssum_in_ta")
    spec["dataset"] = {"data_files": str(p), "format": "json", "json_field": "examples", "split": "test"}
    t = build_task(spec, RunConfig(), ImageConfig())
    d = t.make_doc(t.load_source().rows([0])[0], 0)
    assert d.vars["source"] == "Long English article." and d.gold == ["தமிழ் சுருக்கம்"]
    assert "in Tamil" in t.render(d)


def test_new_mcq_specs(tmp_path):
    def mk(name, rows, **ds):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        spec = TaskRegistry().get(name)
        spec["dataset"] = dict({"path": str(p), "split": "test"}, **ds)
        t = build_task(spec, RunConfig(), ImageConfig())
        src = t.load_source()
        return [t.make_doc(r, i) for i, r in enumerate(src.rows(range(len(src))))]

    off = mk("offensive_codemix_ta", [{"text": "a", "label": 0}, {"text": "b", "label": 3}, {"text": "c", "label": 5}],
             filter={"label": [0, 1, 2, 3, 4]})
    assert [d.gold for d in off] == [0, 1]
    gm = mk("global_mmlu_hi", [{"sample_id": "s1", "question": "q", "option_a": "w", "option_b": "x", "option_c": "y",
                                "option_d": "z", "answer": "C", "cultural_sensitivity_label": "CS",
                                "subject_category": "STEM"}])
    assert gm[0].gold == 2 and gm[0].group["cultural_sensitivity_label"] == "CS"
    imp = mk("indicmmlu_pro_ta", [{"question_id": 1, "question": "q", "options": list("abcdefghij"), "answer": "D"}])
    assert imp[0].gold == 3 and len(imp[0].choices) == 10
