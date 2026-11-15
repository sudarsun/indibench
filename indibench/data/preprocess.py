"""Named row-level preprocessors that a task spec can request via ``dataset.preprocess``.

A preprocessor takes a list of row dicts and returns a new list of row dicts. They run
*before* sampling, so they may expand or drop rows (e.g. flattening SQuAD JSON).
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List

Rows = List[Dict[str, Any]]
PREPROCESSORS: Dict[str, Callable[..., Rows]] = {}


def register(name: str):
    def deco(fn):
        PREPROCESSORS[name] = fn
        return fn

    return deco


@register("squad_flatten")
def squad_flatten(rows: Rows, **_) -> Rows:
    """Flatten SQuAD-format articles ``{title, paragraphs:[{context, qas:[...]}]}`` into QA rows.

    Accepts rows that are already flat (have ``context``/``question``) unchanged.
    """
    if rows and "question" in rows[0] and "context" in rows[0]:
        return rows
    out: Rows = []
    for art in rows:
        paragraphs = art.get("paragraphs") or []
        for p in paragraphs:
            ctx = (p.get("context") or "").strip()
            for qa in p.get("qas") or []:
                answers = qa.get("answers") or []
                texts = [a.get("text", "").strip() for a in answers if isinstance(a, dict)]
                out.append({
                    "id": str(qa.get("id", len(out))),
                    "title": art.get("title", ""),
                    "context": ctx,
                    "question": (qa.get("question") or "").strip(),
                    "answers": {"text": texts,
                                "answer_start": [a.get("answer_start", -1) for a in answers if isinstance(a, dict)]},
                    "is_impossible": bool(qa.get("is_impossible", False)) or not any(texts),
                })
    return out


@register("drop_unanswerable")
def drop_unanswerable(rows: Rows, answers_field: str = "answers", **_) -> Rows:
    def has_answer(r):
        a = r.get(answers_field)
        if isinstance(a, dict):
            a = a.get("text")
        if isinstance(a, str):
            return bool(a.strip())
        return bool(a) and any(str(x).strip() for x in a)

    return [r for r in rows if has_answer(r)]


@register("literal_eval")
def literal_eval_fields(rows: Rows, fields: List[str] = (), **_) -> Rows:
    """Parse stringified Python lists (e.g. MMMU ``options``) into real lists."""
    for r in rows:
        for f in fields:
            v = r.get(f)
            if isinstance(v, str) and v.strip().startswith("["):
                try:
                    r[f] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass
    return rows


@register("require_image")
def require_image(rows: Rows, field: str = "image", **_) -> Rows:
    return [r for r in rows if r.get(field) is not None]


def apply_preprocessors(rows: Rows, specs: Any) -> Rows:
    """``specs`` may be a string, a dict ``{name: kwargs}``, or a list of either."""
    if not specs:
        return rows
    if isinstance(specs, (str, dict)):
        specs = [specs]
    for spec in specs:
        if isinstance(spec, str):
            name, kwargs = spec, {}
        else:
            (name, kwargs), = spec.items()
            kwargs = kwargs or {}
        if name not in PREPROCESSORS:
            raise KeyError(f"Unknown preprocessor {name!r}. Available: {sorted(PREPROCESSORS)}")
        rows = PREPROCESSORS[name](rows, **kwargs)
    return rows
