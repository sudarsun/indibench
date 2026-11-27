"""Answer extraction from free-form generations (MCQ letters, yes/no, numbers)."""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

from indibench.languages import LANGUAGES
from indibench.metrics.normalize import normalize_text

# "Answer" in several Indic languages, so native-instruction prompts still parse.
_ANSWER_WORDS = [
    "final answer", "answer", "ans", "correct option", "correct answer", "option", "choice",
    "விடை", "பதில்", "சரியான விடை", "उत्तर", "सही उत्तर", "జవాబు", "సమాధానం", "ಉತ್ತರ",
    "ഉത്തരം", "উত্তর", "જવાબ", "ਜਵਾਬ", "ଉତ୍ତର", "جواب",
]
_ANS_RE = re.compile(
    r"(?:" + "|".join(re.escape(w) for w in sorted(_ANSWER_WORDS, key=len, reverse=True)) + r")"
    r"\s*(?:is|was|would be|:|：|-|=|–)?\s*(?:option\s*)?[\(\[\*\s]*([A-Za-z])(?![A-Za-z])",
    re.IGNORECASE,
)


def letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def extract_choice(text: str, n_choices: int, choices: Optional[Sequence[str]] = None,
                   lang: str = "") -> Optional[int]:
    """Return the 0-based option index the response selects, or ``None`` if unparseable."""
    if not text:
        return None
    valid = letters(n_choices)
    t = text.strip().replace("**", "")

    # 1) explicit "Answer: X" (take the last such statement — models often reason first)
    found = [m.group(1) for m in _ANS_RE.finditer(t)]
    for cand in reversed(found):
        if cand.upper() in valid and (cand.isupper() or len(t) < 40):
            return valid.index(cand.upper())

    # 2) response begins with the letter: "B", "B.", "(B)", "B) ..."
    m = re.match(r"^\s*[\(\[]?([A-Z])(?:[\)\]\.:,]|\s|$)", t)
    if m and m.group(1) in valid:
        return valid.index(m.group(1))

    # 3) "(X)" anywhere — unique
    paren = {x for x in re.findall(r"\(([A-Z])\)", t) if x in valid}
    if len(paren) == 1:
        return valid.index(paren.pop())

    # 4) the option text itself appears (prefer the longest, unique match)
    if choices:
        nt = normalize_text(t, lang)
        hits = []
        for i, c in enumerate(choices):
            nc = normalize_text(str(c), lang)
            if nc and (nc == nt or (len(nc) >= 2 and nc in nt)):
                hits.append((len(nc), i))
        if hits:
            hits.sort(reverse=True)
            if len(hits) == 1 or hits[0][0] > hits[1][0]:
                return hits[0][1]

    # 5) a lone capital letter token (last occurrence)
    lone = [x for x in re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", t) if x in valid]
    if lone:
        return valid.index(lone[-1])
    return None


_YES = {"yes", "true", "correct", "y"}
_NO = {"no", "false", "incorrect", "n"}
for _l in LANGUAGES.values():
    _YES.update(w.lower() for w in _l.yes)
    _NO.update(w.lower() for w in _l.no)


def extract_yes_no(text: str) -> Optional[str]:
    """First yes/no word in any supported language → ``"yes"``/``"no"``."""
    if not text:
        return None
    toks = normalize_text(text, keep_zwj=True).split()
    for tok in toks:
        if tok in _YES:
            return "yes"
        if tok in _NO:
            return "no"
    # Indic words can carry suffixes (e.g. இல்லை.) — prefix match as a fallback.
    for tok in toks:
        for w in _NO:
            if len(w) > 1 and tok.startswith(w):
                return "no"
        for w in _YES:
            if len(w) > 1 and tok.startswith(w):
                return "yes"
    return None


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:e[-+]?\d+)?|-?\.\d+")


def extract_number(text: str) -> Optional[float]:
    """Last number in the text (commas removed). Native digits are normalised first."""
    if not text:
        return None
    from indibench.metrics.normalize import _native_digits_to_ascii

    t = _native_digits_to_ascii(text)
    nums = _NUM_RE.findall(t)
    for n in reversed(nums):
        try:
            v = float(n.replace(",", "").rstrip("."))
        except (ValueError, OverflowError):
            continue
        # a runaway digit string ("99999…" or "1e999") parses as inf; it is not an answer
        if math.isfinite(v):
            return v
    return None
