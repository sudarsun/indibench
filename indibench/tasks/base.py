"""Task abstraction and the generic, YAML-configurable task types.

A *task spec* (YAML) says where the data lives, how to map its columns, how to prompt
and which metrics to compute. Task *types* implement the behaviour:

  mcq          multiple choice / classification (letter extraction or log-likelihood)
  qa           extractive or short-answer QA scored against reference strings
  generation   translation / summarisation (chrF++, BLEU, ROUGE-L, script fidelity)
  vqa          visual question answering (VQA-acc, ANLS, relaxed acc, yes/no ...)

Special types (``mmmu``, ``mathvista``, ``synthetic_ocr``) live in ``special.py``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from indibench.config import ImageConfig, RunConfig
from indibench.data.loader import RowSource, load_rows
from indibench.languages import LANGUAGES, Language
from indibench.metrics import scores as M
from indibench.metrics.extract import extract_choice, letters
from indibench.models.base import Request
from indibench.tasks import prompts as P
from indibench.utils import SafeDict, resize_to_budget, to_pil

log = logging.getLogger(__name__)

TASK_TYPES: Dict[str, type] = {}


def register_type(name: str):
    def deco(cls):
        TASK_TYPES[name] = cls
        cls.type_name = name
        return cls

    return deco


@dataclass
class Doc:
    id: str
    vars: Dict[str, Any]                       # template variables
    images: List[Any] = field(default_factory=list)
    gold: Any = None                           # task-type specific
    choices: Optional[List[str]] = None
    group: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def _get(row: Dict[str, Any], key: Any) -> Any:
    """Fetch a (possibly dotted) column, e.g. ``answers.text``."""
    if key is None:
        return None
    if isinstance(key, str) and "|" in key:      # "eng_Latn|sentence_eng_Latn": first column present wins
        for alt in key.split("|"):
            v = _get(row, alt.strip())
            if v is not None:
                return v
        return None
    cur: Any = row
    for part in str(key).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, dict) and "text" in v:
        return _as_list(v["text"])
    return [v]


class Task:
    type_name = "base"
    default_metrics: List[str] = []
    needs_images_default = False

    def __init__(self, spec: Dict[str, Any], run: RunConfig, image_cfg: ImageConfig):
        self.spec = spec
        self.run = run
        self.image_cfg = image_cfg
        self.name: str = spec["name"]
        self.family: str = spec.get("family", self.name)
        self.lang: Optional[str] = spec.get("lang")
        self.language: Optional[Language] = LANGUAGES.get(self.lang) if self.lang else None
        self.fields: Dict[str, Any] = spec.get("fields") or {}
        self.prompt_spec: Dict[str, Any] = spec.get("prompt") or {}
        self.metrics: List[str] = list(spec.get("metrics") or self.default_metrics)
        self.max_new_tokens: Optional[int] = spec.get("max_new_tokens")
        self.predict_only: bool = bool(spec.get("predict_only", False))
        gb = spec.get("group_by") or []
        self.group_by: List[str] = [gb] if isinstance(gb, str) else list(gb)
        nf = run.num_fewshot if run.num_fewshot is not None else spec.get("num_fewshot", 0)
        self.num_fewshot: int = int(nf or 0)
        self.instruction_lang = spec.get("instruction_lang") or run.instruction_lang
        self.modality: str = spec.get("modality") or ("image" if self._uses_images() else "text")
        self.root: Optional[str] = None
        self._fewshot_docs: Optional[List[Doc]] = None

    # ------------------------------------------------------------------ helpers
    @property
    def tags(self) -> List[str]:
        return list(self.spec.get("tags") or [])

    @property
    def prompt_lang(self) -> str:
        return self.lang or "en"

    def _uses_images(self) -> bool:
        return bool(self.fields.get("image") or self.fields.get("images") or self.needs_images_default)

    def lang_vars(self) -> Dict[str, Any]:
        v: Dict[str, Any] = dict(self.language.vars()) if self.language else {}
        v.update(self.spec.get("vars") or {})
        return v

    def _truncate(self, text: Any) -> Any:
        lim = self.spec.get("max_input_chars") or self.run.max_input_chars
        if lim and isinstance(text, str) and len(text) > lim:
            return text[:lim]
        return text

    def _images_from_row(self, row: Dict[str, Any]) -> List[Any]:
        keys = self.fields.get("images") or self.fields.get("image")
        if not keys:
            return []
        keys = keys if isinstance(keys, list) else [keys]
        imgs = []
        for k in keys:
            for v in _as_list(_get(row, k)):
                img = to_pil(v, self.root)
                if img is not None:
                    imgs.append(img)
        return imgs

    def prepare_images(self, imgs: List[Any]) -> List[Any]:
        imgs = imgs[: self.image_cfg.max_images]
        return [resize_to_budget(i, self.image_cfg.max_pixels, self.image_cfg.min_pixels) for i in imgs]

    # ------------------------------------------------------------------ data
    def dataset_spec(self) -> Dict[str, Any]:
        ds = dict(self.spec.get("dataset") or {})
        # Per-task `dataset.trust_remote_code` wins over the run-wide default.
        if self.run.trust_remote_code is not None:
            ds.setdefault("trust_remote_code", self.run.trust_remote_code)
        return ds

    def load_source(self, max_rows: Optional[int] = None) -> RowSource:
        ds = self.dataset_spec()
        src = load_rows(ds, ds.get("split"), streaming=self.run.streaming, cache_dir=self.run.cache_dir,
                        token_env=self.run.hf_token_env, max_rows=max_rows)
        self.root = src.root
        return src

    def strata_column(self) -> Optional[str]:
        return self.run.stratify_by or (self.group_by[0] if self.group_by else None)

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        raise NotImplementedError

    def doc_id(self, row: Dict[str, Any], idx: int) -> str:
        key = self.fields.get("id")
        if key and _get(row, key) is not None:
            base = str(_get(row, key))
        else:
            base = str(idx)
        cfg = row.get("_config")
        return f"{cfg}/{base}" if cfg else base

    def doc_group(self, row: Dict[str, Any]) -> Dict[str, str]:
        return {g: str(_get(row, g)) for g in self.group_by if _get(row, g) is not None}

    def load_fewshot(self) -> List[Doc]:
        if self._fewshot_docs is not None:
            return self._fewshot_docs
        self._fewshot_docs = []
        if self.num_fewshot <= 0:
            return self._fewshot_docs
        ds = self.dataset_spec()
        split = self.spec.get("fewshot_split")
        if not split:
            log.warning("%s: num_fewshot=%d but no fewshot_split set; running zero-shot", self.name, self.num_fewshot)
            return self._fewshot_docs
        src = load_rows(ds, split, cache_dir=self.run.cache_dir, token_env=self.run.hf_token_env)
        from indibench.data.loader import sample_indices

        idx = sample_indices(len(src), self.num_fewshot * 3, "random", self.run.seed + 7)
        for i, row in zip(idx, src.rows(idx)):
            d = self.make_doc(row, i)
            if d is not None and not (d.images and not self.spec.get("fewshot_images", False)):
                self._fewshot_docs.append(d)
            if len(self._fewshot_docs) >= self.num_fewshot:
                break
        return self._fewshot_docs

    # ------------------------------------------------------------------ prompting
    def default_template(self) -> str:
        raise NotImplementedError

    def template(self) -> str:
        if self.instruction_lang == "native":
            native = self.prompt_spec.get("template_native") or {}
            if self.prompt_lang in native:
                return native[self.prompt_lang]
        return self.prompt_spec.get("template") or self.default_template()

    def render(self, doc: Doc) -> str:
        v = dict(self.lang_vars())
        v.update(doc.vars)
        lab = P.labels(self.prompt_lang, self.instruction_lang)
        v.update({f"L_{k}": val for k, val in lab.items()})
        return self.template().format_map(SafeDict(v)).strip()

    def gold_text(self, doc: Doc) -> str:
        """Assistant turn used for few-shot demonstrations."""
        return str(doc.gold)

    def build_request(self, doc: Doc, system_prompt: Optional[str] = None) -> Request:
        msgs: List[Dict[str, Any]] = []
        sys_p = self.prompt_spec.get("system") or system_prompt
        if sys_p:
            msgs.append({"role": "system", "content": [{"type": "text", "text": sys_p}]})
        for fs in self.load_fewshot():
            msgs.append({"role": "user", "content": self._content(fs)})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": self.gold_text(fs)}]})
        msgs.append({"role": "user", "content": self._content(doc)})
        return Request(messages=msgs, max_new_tokens=self.max_new_tokens)

    def _content(self, doc: Doc) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"type": "image", "image": im} for im in self.prepare_images(doc.images)]
        parts.append({"type": "text", "text": self.render(doc)})
        return parts

    def wants_loglikelihood(self) -> bool:
        return False

    def build_ll_request(self, doc: Doc, system_prompt: Optional[str] = None) -> Request:
        raise NotImplementedError

    # ------------------------------------------------------------------ scoring
    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        raise NotImplementedError

    def score_ll(self, doc: Doc, lls: List[Tuple[float, int]]) -> Tuple[Any, Dict[str, float]]:
        raise NotImplementedError

    def gold_repr(self, doc: Doc) -> Any:
        return doc.gold

    def corpus_metrics(self, records: List[Dict[str, Any]], n_boot: int) -> Dict[str, Dict[str, Any]]:
        return {}

    def aggregate(self, records: List[Dict[str, Any]], n_boot: int) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        keys: List[str] = []
        for r in records:
            for k in (r.get("metrics") or {}):
                if k not in keys:
                    keys.append(k)
        for k in keys:
            vals = [r["metrics"].get(k) for r in records if r.get("metrics")]
            m, lo, hi = M.mean_ci(vals, n_boot, seed=self.run.seed)
            out[k] = {"value": 100 * m, "ci_low": None if lo is None else 100 * lo,
                      "ci_high": None if hi is None else 100 * hi, "n": len(vals),
                      "higher_is_better": k not in M.LOWER_IS_BETTER}
        out.update(self.corpus_metrics(records, n_boot))
        return out

    def primary_metric(self) -> str:
        return self.spec.get("primary_metric") or (self.metrics[0] if self.metrics else "accuracy")


# ============================================================================ MCQ
@register_type("mcq")
class MultipleChoiceTask(Task):
    """Multiple choice & classification.

    ``fields.choices`` may be a list of columns, one column holding a list, or omitted when
    ``choices`` (static label verbalisers, e.g. ``[Positive, Negative]``) is given in the spec.
    ``fields.answer_format``: index0 | index1 | letter | option_key | text | auto.
    """

    default_metrics = ["accuracy"]

    def default_template(self) -> str:
        instr = P.pick(P.MCQ_INSTRUCTION, self.prompt_lang, self.instruction_lang)
        return "{context_block}{L_question}: {question}\n{choices_block}\n" + instr

    def _choices(self, row: Dict[str, Any]) -> List[str]:
        static = self.spec.get("choices")
        if static:
            return [str(c) for c in static]
        spec = self.fields.get("choices")
        if isinstance(spec, list):
            vals = [_get(row, c) for c in spec]
            return [str(v) for v in vals if v is not None and str(v).strip() not in ("", "nan", "None")]
        return [str(c) for c in _as_list(_get(row, spec))]

    def _resolve_answer(self, val: Any, choices: List[str]) -> Optional[int]:
        fmt = self.fields.get("answer_format", "auto")
        if val is None:
            return None
        s = str(val).strip()
        mapping = self.spec.get("answer_map")
        if mapping:
            s = str(mapping.get(val, mapping.get(s, s)))
            fmt = "text" if not s.lstrip("-").isdigit() else "index0"
        try:
            if fmt == "index0":
                return int(float(s))
            if fmt == "index1":
                return int(float(s)) - 1
            if fmt == "letter":
                return letters(26).index(s.upper()[:1])
            if fmt == "option_key":
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) - 1 if m else None
            if fmt in ("text", "auto"):
                low = [c.strip().lower() for c in choices]
                if s.lower() in low:
                    return low.index(s.lower())
                if fmt == "auto":
                    if len(s) == 1 and s.isalpha():
                        return letters(26).index(s.upper())
                    if s.lstrip("-").isdigit():
                        return int(s)
                return None
        except (ValueError, IndexError):
            return None
        raise ValueError(f"{self.name}: unknown answer_format {fmt!r}")

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        choices = self._choices(row)
        if len(choices) < 2:
            return None
        gold = self._resolve_answer(_get(row, self.fields.get("answer")), choices)
        if gold is not None and gold < 0:     # e.g. hidden test labels (-1)
            gold = None
        if gold is None and not self.predict_only:
            return None
        if gold is not None and not (0 <= gold < len(choices)):
            return None
        L = letters(len(choices))
        v: Dict[str, Any] = {}
        for k, col in self.fields.items():
            if k in ("choices", "answer", "answer_format", "image", "images", "id"):
                continue
            if isinstance(col, str):
                v[k] = self._truncate(_get(row, col))
        lab = P.labels(self.prompt_lang, self.instruction_lang)
        ctx = v.get("context")
        hint = v.get("hint")
        block = ""
        if ctx:
            block += f"{lab['passage']}: {ctx}\n\n"
        if hint:
            block += f"{lab['hint']}: {hint}\n\n"
        v["context_block"] = block
        v.setdefault("question", "")
        v["choices_block"] = "\n".join(f"{l}. {c}" for l, c in zip(L, choices))
        v["choices_inline"] = ", ".join(f"({l}) {c}" for l, c in zip(L, choices))
        return Doc(id=self.doc_id(row, idx), vars=v, images=self._images_from_row(row), gold=gold,
                   choices=choices, group=self.doc_group(row))

    def gold_text(self, doc: Doc) -> str:
        return letters(len(doc.choices or []))[doc.gold]

    def gold_repr(self, doc: Doc) -> Any:
        return None if doc.gold is None else letters(len(doc.choices or []))[doc.gold]

    def wants_loglikelihood(self) -> bool:
        mode = self.run.mcq_mode or self.spec.get("mcq_mode", "generate")
        return mode == "loglikelihood"

    def build_ll_request(self, doc: Doc, system_prompt: Optional[str] = None) -> Request:
        req = self.build_request(doc, system_prompt)
        target = self.spec.get("ll_target", "letter")
        req.continuations = letters(len(doc.choices or [])) if target == "letter" else [f"{c}" for c in doc.choices or []]
        return req

    def _metrics_for(self, pred: Optional[int], doc: Doc) -> Dict[str, float]:
        if self.predict_only or doc.gold is None:
            return {}
        return {"accuracy": float(pred == doc.gold), "parse_rate": float(pred is not None)}

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        idx = extract_choice(response, len(doc.choices or []), doc.choices, self.prompt_lang)
        pred = None if idx is None else letters(len(doc.choices or []))[idx]
        return pred, self._metrics_for(idx, doc)

    def score_ll(self, doc: Doc, lls: List[Tuple[float, int]]) -> Tuple[Any, Dict[str, float]]:
        norm = self.spec.get("ll_target", "letter") != "letter"
        vals = [(lp / max(n, 1)) if norm else lp for lp, n in lls]
        idx = max(range(len(vals)), key=lambda i: vals[i]) if vals else None
        pred = None if idx is None else letters(len(doc.choices or []))[idx]
        m = self._metrics_for(idx, doc)
        m.pop("parse_rate", None)
        return pred, m

    def corpus_metrics(self, records, n_boot):
        if "macro_f1" not in self.metrics:
            return {}
        pairs = [(r.get("pred"), r.get("gold")) for r in records if r.get("gold") is not None]
        if not pairs:
            return {}
        conv = lambda x: None if x is None else ord(str(x)) - 65  # noqa: E731
        return {"macro_f1": {"value": M.macro_f1([conv(p) for p, _ in pairs], [conv(g) for _, g in pairs]),
                             "ci_low": None, "ci_high": None, "n": len(pairs), "higher_is_better": True}}


# ============================================================================ QA
@register_type("qa")
class QATask(Task):
    """Extractive / short-answer QA (optionally with images) against reference strings."""

    default_metrics = ["exact_match", "f1", "char_f1"]

    def default_template(self) -> str:
        if self.fields.get("context"):
            instr = P.pick(P.EXTRACTIVE_QA_INSTRUCTION, self.prompt_lang, self.instruction_lang)
            return instr + "\n\n{L_passage}: {context}\n\n{L_question}: {question}\n{L_answer}:"
        instr = P.pick(P.SHORT_ANSWER_INSTRUCTION, self.prompt_lang, self.instruction_lang)
        return "{question}\n" + instr

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        golds = [str(a).strip() for a in _as_list(_get(row, self.fields.get("answers", "answers"))) if str(a).strip()]
        if not golds and not self.predict_only:
            return None
        v = {}
        for k, col in self.fields.items():
            if k in ("answers", "image", "images", "id") or not isinstance(col, str):
                continue
            v[k] = self._truncate(_get(row, col))
        v.setdefault("question", "")
        return Doc(id=self.doc_id(row, idx), vars=v, images=self._images_from_row(row), gold=golds,
                   group=self.doc_group(row))

    def gold_text(self, doc: Doc) -> str:
        return doc.gold[0] if doc.gold else ""

    def postprocess(self, response: str) -> str:
        r = response.strip()
        # Strip a leading "Answer:" label the model may echo (in any supported language).
        r = re.sub(r"^\s*(answer|விடை|பதில்|उत्तर)\s*[:：]\s*", "", r, flags=re.IGNORECASE)
        if self.spec.get("first_line_only", True):
            r = r.split("\n")[0].strip()
        return r

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        pred = self.postprocess(response)
        if self.predict_only or not doc.gold:
            return pred, {}
        lang = self.spec.get("metric_lang") or self.prompt_lang
        out = {}
        for m in self.metrics:
            if m in M.SAMPLE_METRICS:
                if m == "script_fidelity":
                    out[m] = M.script_fidelity(pred, doc.gold, self.spec.get("target_lang") or lang)
                else:
                    out[m] = M.SAMPLE_METRICS[m](pred, doc.gold, lang)
        return pred, out


# ============================================================================ VQA
@register_type("vqa")
class VQATask(QATask):
    default_metrics = ["vqa_accuracy", "exact_match"]

    def default_template(self) -> str:
        instr = P.pick(P.SHORT_ANSWER_INSTRUCTION, self.prompt_lang, self.instruction_lang)
        return "{question}\n" + instr


# ============================================================================ generation
@register_type("generation")
class GenerationTask(Task):
    """Translation / summarisation / open generation against reference text(s)."""

    default_metrics = ["chrf", "bleu", "script_fidelity"]

    def default_template(self) -> str:
        if self.spec.get("subtype") == "summarization":
            return P.pick(P.SUMMARY_INSTRUCTION, self.prompt_lang, self.instruction_lang) + "\n\n{source}"
        return P.TRANSLATION_INSTRUCTION["en"] + "\n\n{source}"

    def lang_vars(self) -> Dict[str, Any]:
        v = super().lang_vars()
        src = LANGUAGES.get(self.spec.get("source_lang") or "")
        tgt = LANGUAGES.get(self.spec.get("target_lang") or "")
        if src:
            v.update(src_name=src.name, src_native=src.native, src_iso=src.iso)
        if tgt:
            v.update(tgt_name=tgt.name, tgt_native=tgt.native, tgt_iso=tgt.iso)
        return v

    @property
    def prompt_lang(self) -> str:
        return self.spec.get("target_lang") or self.lang or "en"

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        src = _get(row, self.fields.get("source", "source"))
        refs = [str(x) for x in _as_list(_get(row, self.fields.get("target", "target"))) if str(x).strip()]
        if not src or (not refs and not self.predict_only):
            return None
        v = {"source": self._truncate(str(src))}
        for k, col in self.fields.items():
            if k not in ("source", "target", "image", "images", "id") and isinstance(col, str):
                v[k] = self._truncate(_get(row, col))
        return Doc(id=self.doc_id(row, idx), vars=v, images=self._images_from_row(row), gold=refs,
                   group=self.doc_group(row))

    def gold_text(self, doc: Doc) -> str:
        return doc.gold[0]

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        pred = response.strip()
        if self.spec.get("first_line_only", self.spec.get("subtype") != "summarization"):
            pred = pred.split("\n")[0].strip() if pred else pred
        if self.predict_only or not doc.gold:
            return pred, {}
        lang = self.prompt_lang
        out: Dict[str, float] = {}
        for m in self.metrics:
            if m in M.SAMPLE_METRICS:
                out[m] = M.SAMPLE_METRICS[m](pred, doc.gold, lang)
        return pred, out

    def corpus_metrics(self, records, n_boot):
        recs = [r for r in records if r.get("gold")]
        if not recs:
            return {}
        preds = [r.get("pred") or "" for r in recs]
        refs = [r["gold"][0] for r in recs]
        nb = min(n_boot, 200)
        out = {}
        if "chrf" in self.metrics:
            s, lo, hi = M.corpus_ci(lambda p, r: M.corpus_chrf(p, r, 2), preds, refs, nb, self.run.seed)
            out["chrf"] = {"value": s, "ci_low": lo, "ci_high": hi, "n": len(recs), "higher_is_better": True}
        if "bleu" in self.metrics:
            s, lo, hi = M.corpus_ci(M.corpus_bleu, preds, refs, nb, self.run.seed)
            out["bleu"] = {"value": s, "ci_low": lo, "ci_high": hi, "n": len(recs), "higher_is_better": True}
        return out


def build_task(spec: Dict[str, Any], run: RunConfig, image_cfg: ImageConfig) -> Task:
    import indibench.tasks.special  # noqa: F401  (registers special types)

    t = spec.get("type")
    if t not in TASK_TYPES:
        raise KeyError(f"Task {spec.get('name')}: unknown type {t!r}. Available: {sorted(TASK_TYPES)}")
    return TASK_TYPES[t](spec, run, image_cfg)
