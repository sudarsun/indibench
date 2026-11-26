"""Tests for the math / code / JEEBench task types, MCQ extensions and the new specs."""
import json

import pytest

from indibench.config import ImageConfig, RunConfig, load_config
from indibench.tasks.base import build_task
from indibench.tasks.reasoning import (extract_code, extract_math_answer, jee_letters, jee_score, last_boxed,
                                       math_equal, run_python)
from indibench.tasks.registry import TaskRegistry


# ------------------------------------------------------------------ math
@pytest.mark.parametrize("pred,gold,ok", [
    ("\\frac{1}{2}", "0.5", True), ("\\dfrac12", "\\frac{1}{2}", True), ("1,000", "1000", True),
    ("5.0", "5", True), ("x = 3", "3", True), ("90^\\circ", "90", True), ("\\text{(B)}", "(B)", True),
    ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})", True), ("2\\sqrt{3}", "\\sqrt{12}", True),
    ("7", "8", False), ("", "1", False), ("10\\%", "10", True),
])
def test_math_equal(pred, gold, ok):
    assert math_equal(pred, gold) is ok


def test_math_extraction():
    assert last_boxed("so \\boxed{\\frac{a}{b}} and") == "\\frac{a}{b}"
    assert extract_math_answer("Thus x=2.\nThe final answer is $42$.") == "42"
    assert extract_math_answer("Step 1 ... #### 1,234") == "1,234"
    assert extract_math_answer("we get 3 apples and then 17") == "17"
    assert extract_math_answer("<long>... \\boxed{12} hmm \\boxed{13}") == "13"


# ------------------------------------------------------------------ code execution
def test_run_python_pass_fail_timeout():
    assert run_python("assert 1 + 1 == 2")[0]
    ok, err = run_python("assert 1 == 2")
    assert not ok and "AssertionError" in err
    assert run_python("while True: pass", timeout=1.5) == (False, "timeout")


def test_extract_code():
    assert extract_code("text\n```python\ndef f():\n    return 1\n```\nmore") == "def f():\n    return 1"
    assert extract_code("def f(): return 1") == "def f(): return 1"


def _code_task(subtype, tmp_path, allow=True):
    rows = ([{"task_id": "HE/0", "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
              "test": "def check(f):\n    assert f(2, 3) == 5\n", "entry_point": "add"}]
            if subtype == "humaneval" else
            [{"task_id": 1, "prompt": "Write a function to add two numbers.",
              "test_list": ["assert add(2, 3) == 5"], "test_imports": []}])
    p = tmp_path / f"{subtype}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    spec = TaskRegistry().get("humaneval" if subtype == "humaneval" else "mbpp")
    spec["dataset"] = {"path": str(p), "split": "test"}
    return build_task(spec, RunConfig(allow_code_execution=allow), ImageConfig())


@pytest.mark.parametrize("subtype", ["humaneval", "mbpp"])
def test_code_tasks_execute(subtype, tmp_path):
    task = _code_task(subtype, tmp_path)
    doc = task.make_doc(task.load_source().rows([0])[0], 0)
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert task.score(doc, good)[1]["pass@1"] == 1.0
    assert task.score(doc, bad)[1]["pass@1"] == 0.0
    if subtype == "humaneval":                       # body-only completion is also accepted
        assert task.score(doc, "    return a + b\n")[1]["pass@1"] == 1.0


def test_code_execution_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("INDIBENCH_ALLOW_CODE_EXECUTION", raising=False)
    task = _code_task("mbpp", tmp_path, allow=False)
    with pytest.raises(PermissionError, match="allow-code-execution"):
        task.load_source()


# ------------------------------------------------------------------ JEEBench
def test_jee_scoring_rules():
    assert jee_score("MCQ(multiple)", "AD", "AD") == 1.0
    assert jee_score("MCQ(multiple)", "A", "AD") == 0.25           # partial, no wrong option
    assert jee_score("MCQ(multiple)", "AB", "AD") == 0.0            # any wrong option -> 0
    assert jee_score("Numeric", "2.345", "2.34") == 1.0
    assert jee_score("Numeric", "2.40", "2.34") == 0.0
    assert jee_score("Integer", "7", "7") == 1.0
    assert jee_letters("... so options (B) and (D) hold.\nFinal Answer: BD") == "BD"
    assert jee_letters("Final Answer: \\boxed{ACD}") == "ACD"


def test_jeebench_task_end_to_end(tmp_path):
    rows = [{"subject": "phy", "description": "JEE Adv 2016 Paper 1", "gold": "AD", "index": 9,
             "type": "MCQ(multiple)", "question": "Q1 (A) a (B) b (C) c (D) d"},
            {"subject": "math", "description": "JEE Adv 2017 Paper 2", "gold": "3.14", "index": 3,
             "type": "Numeric", "question": "Compute pi"}]
    p = tmp_path / "jee.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    spec = TaskRegistry().get("jeebench")
    spec["dataset"] = {"path": str(p), "split": "test"}
    task = build_task(spec, RunConfig(), ImageConfig())
    docs = [task.make_doc(r, i) for i, r in enumerate(task.load_source().rows([0, 1]))]
    assert "multiple options can be correct" in task.render(docs[0])
    assert task.score(docs[0], "reasoning\nFinal Answer: A")[1]["score"] == 0.25
    assert task.score(docs[1], "Final Answer: 3.141")[1]["exact"] == 1.0


# ------------------------------------------------------------------ MCQ extensions
def _mcq(spec_name, rows, tmp_path, **over):
    p = tmp_path / f"{spec_name}.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    spec = TaskRegistry().get(spec_name)
    spec["dataset"] = {"path": str(p), "split": "test", **over}
    task = build_task(spec, RunConfig(), ImageConfig())
    src = task.load_source()
    return task, [task.make_doc(r, i) for i, r in enumerate(src.rows(range(len(src))))]


def test_gpqa_gold_index_and_shuffle(tmp_path):
    rows = [{"Question": f"q{i}", "Correct Answer": "RIGHT", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"} for i in range(20)]
    task, docs = _mcq("gpqa_diamond", rows, tmp_path)
    assert all(d.choices[d.gold] == "RIGHT" for d in docs)
    assert len({d.gold for d in docs}) > 1                          # correct answer not always "A"
    _, again = _mcq("gpqa_diamond", rows, tmp_path)
    assert [d.gold for d in docs] == [d.gold for d in again]       # deterministic


def test_truthfulqa_onehot_and_arc_label(tmp_path):
    _, docs = _mcq("truthfulqa_mc1", [{"question": "q", "mc1_targets": {"choices": ["t", "f1", "f2"],
                                                                         "labels": [1, 0, 0]}}], tmp_path)
    assert docs[0].choices[docs[0].gold] == "t"
    _, docs = _mcq("arc_challenge", [{"id": "x", "question": "q", "answerKey": "3",
                                      "choices": {"text": ["a", "b", "c", "d"], "label": ["1", "2", "3", "4"]}}],
                   tmp_path)
    assert docs[0].gold == 2


def test_include_and_tanglish_filter(tmp_path):
    _, docs = _mcq("include_ta", [{"question": "வினா", "option_a": "அ", "option_b": "ஆ", "option_c": "இ",
                                   "option_d": "ஈ", "answer": 2, "domain": "STEM", "subject": "Physics"}], tmp_path)
    assert docs[0].gold == 2 and docs[0].group == {"domain": "STEM", "subject": "Physics"}
    _, docs = _mcq("include_ta", [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1}], tmp_path)
    assert docs[0].choices == ["a", "b", "c", "d"]                 # choices_fallback (older schema)
    rows = [{"text": "Semma padam", "label": 0}, {"text": "mokka", "label": 1}, {"text": "x", "label": 4}]
    _, docs = _mcq("tanglish_sentiment", rows, tmp_path, filter={"label": [0, 1, 2]})
    assert [d.gold for d in docs] == [0, 1]


def test_boolq_answer_map(tmp_path):
    rows = [{"question": "q", "passage": "p", "answer": True}, {"question": "q2", "passage": "p", "answer": False}]
    _, docs = _mcq("boolq_indic_ta", rows, tmp_path)
    assert [d.gold for d in docs] == [0, 1]


def test_indicgenbench_examples_container(tmp_path):
    p = tmp_path / "xquad_ta_test.json"
    p.write_text(json.dumps({"examples": [{"id": "1", "lang": "ta", "context": "சென்னை தலைநகரம்.",
                                           "question": "தலைநகரம்?", "answers": [{"text": "சென்னை"}]}]},
                            ensure_ascii=False), encoding="utf-8")
    spec = TaskRegistry().get("xquad_in_ta")
    spec["dataset"] = {"data_files": str(p), "format": "json", "json_field": "examples", "split": "test"}
    task = build_task(spec, RunConfig(), ImageConfig())
    doc = task.make_doc(task.load_source().rows([0])[0], 0)
    assert doc.gold == ["சென்னை"]


# ------------------------------------------------------------------ registry / selection
def test_new_groups_and_tags():
    reg = TaskRegistry()
    assert set(reg.resolve(["code"])) == {"humaneval", "mbpp"}
    assert "jeebench" in reg.resolve(["tag:indian_english"]) and "milu_en" in reg.resolve(["tag:indian_english"])
    assert "humaneval" not in reg.resolve(["full"])
    ta = reg.resolve(["tamil_extended"], ["ta"])
    assert {"include_ta", "mmlu_indic_ta", "xquad_in_ta", "tanglish_sentiment"} <= set(ta)


def test_math_run_with_dummy_oracle(tmp_path):
    from indibench.runner import Evaluator

    p = tmp_path / "gsm.jsonl"
    p.write_text("\n".join(json.dumps({"question": f"What is {i}+{i}?", "answer": f"work... #### {2 * i}"})
                           for i in range(5)))
    spec_file = tmp_path / "t.yaml"
    spec_file.write_text(f"""
tasks:
  - name: local_gsm
    type: math
    dataset: {{path: "{p}", split: test}}
    answer_after: "####"
    fields: {{question: question, answer: answer}}
""")
    cfg = load_config(None, {"model.backend": "dummy", "run.tasks": ["local_gsm"], "run.task_files": [str(spec_file)],
                             "run.output_dir": str(tmp_path), "run.run_name": "m"}, ["model.extra.mode=oracle"])
    res = Evaluator(cfg).run()["results"]["local_gsm"]
    assert res["metrics"]["accuracy"]["value"] == 100.0
