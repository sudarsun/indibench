"""Per-sample and corpus-level metrics.

Sample metrics have the signature ``fn(pred: str, golds: list[str], lang: str) -> float``
and are averaged (with bootstrap CIs). Corpus metrics take aligned lists and return one
number (chrF++, BLEU, macro-F1).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

from indibench.languages import LANGUAGES, script_ratio
from indibench.metrics.extract import extract_number, extract_yes_no
from indibench.metrics.normalize import graphemes, normalize_text, tokenize


def _max_over(fn, pred: str, golds: Sequence[str], lang: str) -> float:
    golds = [g for g in golds if g is not None] or [""]
    return max(fn(pred, str(g), lang) for g in golds)


# --------------------------------------------------------------------------- string match
def _em(pred: str, gold: str, lang: str) -> float:
    return float(normalize_text(pred, lang) == normalize_text(gold, lang))


def _token_f1(pred: str, gold: str, lang: str) -> float:
    p, g = tokenize(pred, lang), tokenize(gold, lang)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _char_f1(pred: str, gold: str, lang: str) -> float:
    """Character-multiset F1 — robust to agglutinative suffixes (சென்னையில் vs சென்னை)."""
    p = [c for c in normalize_text(pred, lang) if not c.isspace()]
    g = [c for c in normalize_text(gold, lang) if not c.isspace()]
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _contains(pred: str, gold: str, lang: str) -> float:
    g = normalize_text(gold, lang)
    return float(bool(g) and g in normalize_text(pred, lang))


def _levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _anls_single(pred: str, gold: str, lang: str, tau: float = 0.5) -> float:
    p = " ".join(str(pred).strip().lower().split())
    g = " ".join(str(gold).strip().lower().split())
    if not p and not g:
        return 1.0
    nl = _levenshtein(p, g) / max(len(p), len(g), 1)
    return 1.0 - nl if nl < tau else 0.0


def _cer(pred: str, gold: str, lang: str) -> float:
    """Grapheme-cluster error rate (lower is better); NFC + whitespace-normalised."""
    import unicodedata

    p = graphemes(" ".join(unicodedata.normalize("NFC", pred).split()))
    g = graphemes(" ".join(unicodedata.normalize("NFC", gold).split()))
    if not g:
        return float(bool(p))
    return min(1.0, _levenshtein(p, g) / len(g))


def _relaxed(pred: str, gold: str, lang: str, tol: float = 0.05) -> float:
    """ChartQA relaxed accuracy: numeric within 5 % or exact string match."""
    gp, pp = _to_float(gold), extract_number(pred)
    if gp is not None and pp is not None:
        if gp == 0:
            return float(pp == 0)
        return float(abs(pp - gp) / abs(gp) <= tol)
    return _em(pred, gold, lang)


def _to_float(s: str) -> Optional[float]:
    try:
        v = float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _rouge_l(pred: str, gold: str, lang: str) -> float:
    """ROUGE-L F1 on Unicode-aware tokens (``rouge_score`` drops non-ASCII characters!)."""
    p, g = tokenize(pred, lang), tokenize(gold, lang)
    if not p or not g:
        return 0.0
    # LCS
    prev = [0] * (len(g) + 1)
    for tp in p:
        cur = [0]
        for j, tg in enumerate(g, 1):
            cur.append(prev[j - 1] + 1 if tp == tg else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- list-gold metrics
def exact_match(pred, golds, lang=""):
    return _max_over(_em, pred, golds, lang)


def f1(pred, golds, lang=""):
    return _max_over(_token_f1, pred, golds, lang)


def char_f1(pred, golds, lang=""):
    return _max_over(_char_f1, pred, golds, lang)


def contains(pred, golds, lang=""):
    return _max_over(_contains, pred, golds, lang)


def anls(pred, golds, lang=""):
    return _max_over(_anls_single, pred, golds, lang)


def relaxed_accuracy(pred, golds, lang=""):
    return _max_over(_relaxed, pred, golds, lang)


def rouge_l(pred, golds, lang=""):
    return _max_over(_rouge_l, pred, golds, lang)


def cer(pred, golds, lang=""):
    golds = [g for g in golds if g is not None] or [""]
    return min(_cer(pred, str(g), lang) for g in golds)


def vqa_accuracy(pred, golds, lang=""):
    """VQAv2/TextVQA soft accuracy: min(#annotators agreeing / 3, 1), averaged over 10-choose-9."""
    p = normalize_text(pred, lang)
    g = [normalize_text(x, lang) for x in golds]
    if len(g) < 2:
        return float(bool(g) and p == g[0])
    accs = []
    for i in range(len(g)):
        others = g[:i] + g[i + 1:]
        accs.append(min(1.0, sum(o == p for o in others) / 3.0))
    return sum(accs) / len(accs)


def yesno_accuracy(pred, golds, lang=""):
    want = extract_yes_no(str(golds[0])) if golds else None
    got = extract_yes_no(pred)
    return float(want is not None and got == want)


def script_fidelity(pred, golds, lang=""):
    """Share of letters written in the target language's script (1.0 = fully in-script)."""
    if lang not in LANGUAGES:
        return float("nan")
    return script_ratio(pred, LANGUAGES[lang])


SAMPLE_METRICS: Dict[str, Callable[[str, List[str], str], float]] = {
    "exact_match": exact_match,
    "f1": f1,
    "char_f1": char_f1,
    "contains": contains,
    "anls": anls,
    "relaxed_accuracy": relaxed_accuracy,
    "rouge_l": rouge_l,
    "cer": cer,
    "vqa_accuracy": vqa_accuracy,
    "yesno_accuracy": yesno_accuracy,
    "script_fidelity": script_fidelity,
}

LOWER_IS_BETTER = {"cer"}


# --------------------------------------------------------------------------- corpus metrics
def corpus_chrf(preds: List[str], refs: List[str], word_order: int = 2) -> float:
    import sacrebleu

    return sacrebleu.corpus_chrf(preds, [refs], word_order=word_order).score


def corpus_bleu(preds: List[str], refs: List[str], lang: str = "") -> float:
    import sacrebleu

    # 'flores200' SentencePiece tokenisation is the standard for Indic MT; fall back to 13a.
    for tok in ("flores200", "13a"):
        try:
            return sacrebleu.corpus_bleu(preds, [refs], tokenize=tok).score
        except Exception:
            continue
    return float("nan")


def macro_f1(pred_labels: List[Optional[int]], gold_labels: List[int]) -> float:
    classes = sorted(set(gold_labels))
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != c and g == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return 100.0 * sum(f1s) / len(f1s) if f1s else float("nan")


# --------------------------------------------------------------------------- aggregation
def mean_ci(values: List[float], n_boot: int = 1000, seed: int = 0):
    """Mean and percentile-bootstrap 95 % CI, ignoring NaNs."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), None, None
    m = sum(vals) / len(vals)
    if n_boot <= 0 or len(vals) < 2:
        return m, None, None
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]


def corpus_ci(fn: Callable[[List[str], List[str]], float], preds: List[str], refs: List[str],
              n_boot: int = 200, seed: int = 0):
    score = fn(preds, refs)
    if n_boot <= 0 or len(preds) < 2:
        return score, None, None
    rng = random.Random(seed)
    n = len(preds)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(fn([preds[i] for i in idx], [refs[i] for i in idx]))
    boots.sort()
    return score, boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]
