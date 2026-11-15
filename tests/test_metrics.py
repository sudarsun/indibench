import math

from indibench.data.loader import sample_indices
from indibench.languages import LANGUAGES, script_ratio
from indibench.metrics import scores as M
from indibench.metrics.extract import extract_choice, extract_number, extract_yes_no
from indibench.metrics.normalize import graphemes, normalize_text
from indibench.utils import strip_thinking


def test_normalize_indic():
    assert normalize_text("சென்னை।") == "சென்னை"
    assert normalize_text("௧௨௩") == "123"            # Tamil digits
    assert normalize_text("१२") == "12"               # Devanagari digits
    assert normalize_text("ab‍c") == "abc"       # ZWJ removed
    assert normalize_text("The Cat!") == "cat"


def test_char_f1_agglutinative():
    assert M.char_f1("சென்னையில்", ["சென்னை"], "ta") > 0.7
    assert M.f1("சென்னையில்", ["சென்னை"], "ta") == 0.0


def test_extract_choice_variants():
    ch = ["மதுரை", "சென்னை", "கோவை", "சேலம்"]
    assert extract_choice("B", 4) == 1
    assert extract_choice("(C) கோவை", 4) == 2
    assert extract_choice("Let me think... The answer is D.", 4) == 3
    assert extract_choice("விடை: A", 4) == 0
    assert extract_choice("சரியான விடை சென்னை ஆகும்", 4, ch, "ta") == 1
    assert extract_choice("I am not sure", 4) is None
    assert extract_choice("**Answer:** B", 4) == 1


def test_yes_no_multilingual():
    assert extract_yes_no("ஆம்") == "yes"
    assert extract_yes_no("இல்லை.") == "no"
    assert extract_yes_no("नहीं") == "no"
    assert extract_yes_no("Yes, it is.") == "yes"
    assert extract_yes_no("అవును") == "yes"


def test_vqa_anls_relaxed_cer():
    assert M.vqa_accuracy("cat", ["cat"] * 10) == 1.0
    assert 0 < M.vqa_accuracy("cat", ["cat"] * 2 + ["dog"] * 8) < 1
    assert M.anls("hello world", ["hello world"]) == 1.0
    assert M.anls("xyz", ["hello world"]) == 0.0
    assert M.relaxed_accuracy("The value is 102", ["100"]) == 1.0
    assert M.relaxed_accuracy("120", ["100"]) == 0.0
    assert M.cer("சென்னை", ["சென்னை"]) == 0.0
    assert 0 < M.cer("சென்ன", ["சென்னை"]) < 1
    assert len(graphemes("தமிழ்")) == 3  # த, மி, ழ்


def test_script_fidelity_and_numbers():
    ta = LANGUAGES["ta"]
    assert script_ratio("இன்று வானிலை", ta) == 1.0
    assert script_ratio("hello", ta) == 0.0
    assert math.isclose(script_ratio("ஆம் yes", ta), 3 / 6, rel_tol=0.01) or script_ratio("ஆம் yes", ta) > 0
    assert extract_number("the answer is 1,234.5") == 1234.5
    assert extract_number("௪௨") == 42.0


def test_strip_thinking():
    assert strip_thinking("<think>blah</think> B") == "B"
    assert strip_thinking("<think>never closed") == ""
    assert strip_thinking("reasoning</think>A") == "A"


def test_sampling_deterministic_and_stratified():
    a = sample_indices(100, 10, "random", 1)
    assert a == sample_indices(100, 10, "random", 1) and len(a) == 10
    assert sample_indices(100, 0.25, "first", 1) == list(range(25))
    strata = ["x"] * 90 + ["y"] * 10
    s = sample_indices(100, 10, "stratified", 3, strata)
    assert sum(1 for i in s if strata[i] == "y") == 5


def test_corpus_metrics():
    assert M.corpus_chrf(["இன்று வானிலை"], ["இன்று வானிலை"]) > 99
    assert M.macro_f1([0, 1, 1, 0], [0, 1, 0, 0]) > 0
