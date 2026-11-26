"""IFEval: verifiable instruction following (Zhou et al., 2023), re-implemented without nltk/langdetect.

Each prompt carries 1-3 instructions (``instruction_id_list``) with arguments (``kwargs``).
A deterministic checker verifies every instruction. Metrics follow the paper:

* ``prompt_strict`` / ``prompt_loose`` — share of prompts where *all* instructions pass
* ``inst_strict`` / ``inst_loose``     — share of individual instructions that pass (micro-average)

"Loose" also accepts the response after removing markdown asterisks and/or its first/last line
(models often add "Sure, here is..." wrappers). Instructions with no checker are skipped and
reported as ``coverage``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from indibench.languages import LANGUAGES, script_ratio
from indibench.tasks.base import Doc, Task, _get, register_type


def _cmp(n: int, relation: Optional[str], target: Optional[int]) -> bool:
    if target is None:
        return True
    rel = (relation or "at least").lower().strip()
    return n < target if rel == "less than" else n >= target


def _words(t: str) -> List[str]:
    return re.findall(r"\b\w+(?:['’-]\w+)*\b", t)


def _sentences(t: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", t.strip())
    return [p for p in parts if re.search(r"\w", p)]


def _strip_fences(t: str) -> str:
    t = t.strip()
    t = re.sub(r"^```(?:json|JSON|Json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t).strip()


def _postscript(t: str, marker: str) -> bool:
    m = (marker or "P.S.").lower().strip()
    if m == "p.p.s":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif m == "p.s.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(m) + r".*$"
    return bool(re.findall(pat, t.lower(), flags=re.MULTILINE))


def _paragraphs_stars(t: str, n: int) -> bool:
    paras = re.split(r"\s?\*\*\*\s?", t)
    count = len(paras)
    for i, p in enumerate(paras):
        if not p.strip():
            if i in (0, len(paras) - 1):
                count -= 1
            else:
                return False
    return count == n


def _nth_paragraph(t: str, num: int, nth: int, first_word: str) -> bool:
    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(paras) != num or not (1 <= nth <= len(paras)):
        return False
    words = paras[nth - 1].strip().split()
    if not words:
        return False
    w = re.sub(r"^[^\w]+|[^\w]+$", "", words[0]).lower()
    return w == (first_word or "").strip().lower()


def _response_language(t: str, lang: str) -> Optional[bool]:
    lang = (lang or "").lower()
    if lang in LANGUAGES and LANGUAGES[lang].script != "Latn":
        return script_ratio(t, LANGUAGES[lang]) >= 0.6
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    ascii_share = sum(c.isascii() for c in letters) / len(letters)
    if lang == "en":
        return ascii_share >= 0.9
    try:  # other Latin-script languages need a language identifier
        from langdetect import detect  # type: ignore

        return detect(t) == lang
    except Exception:
        return None


def _bullets(t: str) -> int:
    star = re.findall(r"^\s*\*[^\*].*$", t, flags=re.MULTILINE)
    dash = re.findall(r"^\s*-.*$", t, flags=re.MULTILINE)
    return len(star) + len(dash)


def _highlights(t: str) -> int:
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", t) if h.strip("*").strip())
    n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", t) if h.strip("*").strip())
    return n


def _sections(t: str, splitter: str, n: int) -> bool:
    parts = re.split(r"\s?" + re.escape(splitter or "Section") + r"\s?\d+\s?", t)
    return len(parts) - 1 >= int(n or 1)


def _two_responses(t: str) -> bool:
    parts = [p.strip() for p in t.split("******")]
    parts = [p for p in parts if p]
    return len(parts) == 2 and parts[0] != parts[1]


def _json_ok(t: str) -> bool:
    try:
        json.loads(_strip_fences(t))
        return True
    except ValueError:
        return False


Checker = Callable[[str, Dict[str, Any], str], Optional[bool]]

CHECKERS: Dict[str, Checker] = {
    "punctuation:no_comma": lambda t, k, p: "," not in t,
    "detectable_format:title": lambda t, k, p: bool(re.search(r"<<[^\n]+>>", t)),
    "detectable_format:json_format": lambda t, k, p: _json_ok(t),
    "detectable_format:number_bullet_lists": lambda t, k, p: _bullets(t) == int(k.get("num_bullets") or 0),
    "detectable_format:number_highlighted_sections":
        lambda t, k, p: _highlights(t) >= int(k.get("num_highlights") or 1),
    "detectable_format:multiple_sections":
        lambda t, k, p: _sections(t, k.get("section_spliter"), k.get("num_sections")),
    "detectable_format:constrained_response":
        lambda t, k, p: any(o in t for o in ("My answer is yes.", "My answer is no.", "My answer is maybe.")),
    "detectable_content:number_placeholders":
        lambda t, k, p: len(re.findall(r"\[.*?\]", t)) >= int(k.get("num_placeholders") or 1),
    "detectable_content:postscript": lambda t, k, p: _postscript(t, k.get("postscript_marker")),
    "length_constraints:number_words":
        lambda t, k, p: _cmp(len(_words(t)), k.get("relation"), k.get("num_words")),
    "length_constraints:number_sentences":
        lambda t, k, p: _cmp(len(_sentences(t)), k.get("relation"), k.get("num_sentences")),
    "length_constraints:number_paragraphs": lambda t, k, p: _paragraphs_stars(t, int(k.get("num_paragraphs") or 1)),
    "length_constraints:nth_paragraph_first_word":
        lambda t, k, p: _nth_paragraph(t, int(k.get("num_paragraphs") or 1), int(k.get("nth_paragraph") or 1),
                                       k.get("first_word") or ""),
    "keywords:existence":
        lambda t, k, p: all(re.search(re.escape(w), t, flags=re.IGNORECASE) for w in (k.get("keywords") or [])),
    "keywords:frequency":
        lambda t, k, p: _cmp(len(re.findall(re.escape(k.get("keyword") or ""), t, flags=re.IGNORECASE)),
                             k.get("relation"), k.get("frequency")),
    "keywords:forbidden_words":
        lambda t, k, p: not any(re.search(r"\b" + re.escape(w) + r"\b", t, flags=re.IGNORECASE)
                                for w in (k.get("forbidden_words") or [])),
    "keywords:letter_frequency":
        lambda t, k, p: _cmp(t.lower().count((k.get("letter") or "").lower()), k.get("let_relation"),
                             k.get("let_frequency")),
    "language:response_language": lambda t, k, p: _response_language(t, k.get("language")),
    "change_case:capital_word_frequency":
        lambda t, k, p: _cmp(sum(1 for w in _words(t) if w.isupper() and len(w) > 1), k.get("capital_relation"),
                             k.get("capital_frequency")),
    "change_case:english_capital": lambda t, k, p: t.isupper() and _response_language(t, "en") is True,
    "change_case:english_lowercase": lambda t, k, p: t.islower() and _response_language(t, "en") is True,
    "startend:end_checker":
        lambda t, k, p: t.strip().strip("\"'").lower().endswith((k.get("end_phrase") or "").strip().lower()),
    "startend:quotation": lambda t, k, p: len(t.strip()) > 1 and t.strip()[0] == '"' and t.strip()[-1] == '"',
    "combination:two_responses": lambda t, k, p: _two_responses(t),
    "combination:repeat_prompt":
        lambda t, k, p: t.strip().lower().startswith((k.get("prompt_to_repeat") or "").strip().lower()),
}


def _loose_variants(t: str) -> List[str]:
    lines = t.split("\n")
    bases = [t, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    return [v for b in bases for v in (b, b.replace("*", ""))]


def check_instructions(response: str, ids: List[str], kwargs: List[Dict[str, Any]], prompt: str = ""
                       ) -> Tuple[List[Optional[bool]], List[Optional[bool]]]:
    strict, loose = [], []
    variants = _loose_variants(response or "")
    for i, iid in enumerate(ids):
        fn = CHECKERS.get(iid)
        kw = {k: v for k, v in (kwargs[i] if i < len(kwargs) and kwargs[i] else {}).items() if v is not None}
        if fn is None or not (response or "").strip():
            strict.append(None if fn is None else False)
            loose.append(None if fn is None else False)
            continue
        try:
            s = fn(response, kw, prompt)
            l_ = any(fn(v, kw, prompt) for v in variants if v)
        except Exception:
            s, l_ = False, False
        strict.append(s)
        loose.append(None if s is None else bool(l_))
    return strict, loose


@register_type("ifeval")
class IFEvalTask(Task):
    default_metrics = ["prompt_strict", "prompt_loose"]

    def default_template(self) -> str:
        return "{prompt}"

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        prompt = _get(row, self.fields.get("prompt", "prompt"))
        ids = list(row.get("instruction_id_list") or [])
        if not prompt or not ids:
            return None
        kwargs = [dict(k) if isinstance(k, dict) else {} for k in (row.get("kwargs") or [])]
        return Doc(id=self.doc_id(row, idx), vars={"prompt": str(prompt)}, gold=ids,
                   meta={"ids": ids, "kwargs": kwargs}, group=self.doc_group(row))

    def gold_repr(self, doc: Doc) -> Any:
        return doc.meta["ids"]

    def gold_text(self, doc: Doc) -> str:
        return ""

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        strict, loose = check_instructions(response, doc.meta["ids"], doc.meta["kwargs"], doc.vars["prompt"])
        known_s = [x for x in strict if x is not None]
        known_l = [x for x in loose if x is not None]
        if not known_s:
            return {"unchecked": doc.meta["ids"]}, {}
        pred = {iid: s for iid, s in zip(doc.meta["ids"], strict)}
        return pred, {"prompt_strict": float(all(known_s)), "prompt_loose": float(all(known_l)),
                      "_inst_strict": known_s, "_inst_loose": known_l,
                      "coverage": len(known_s) / len(strict)}

    def aggregate(self, records, n_boot):
        # pull list-valued per-instruction results out before the generic mean aggregation
        inst_s = [x for r in records for x in (r.get("metrics") or {}).get("_inst_strict", [])]
        inst_l = [x for r in records for x in (r.get("metrics") or {}).get("_inst_loose", [])]
        clean = [dict(r, metrics={k: v for k, v in (r.get("metrics") or {}).items() if not k.startswith("_")})
                 for r in records]
        out = super().aggregate(clean, n_boot)
        for name, vals in (("inst_strict", inst_s), ("inst_loose", inst_l)):
            if vals:
                out[name] = {"value": 100 * sum(vals) / len(vals), "ci_low": None, "ci_high": None,
                             "n": len(vals), "higher_is_better": True}
        return out
