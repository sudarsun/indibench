"""Task types needing bespoke logic: MMMU, MathVista and synthetic Indic-script OCR."""
from __future__ import annotations

import ast
import glob
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from indibench.data.loader import RowSource
from indibench.languages import LANGUAGES
from indibench.metrics import scores as M
from indibench.metrics.extract import extract_choice, extract_number, letters
from indibench.metrics.normalize import normalize_text
from indibench.tasks import prompts as P
from indibench.tasks.base import Doc, Task, _get, register_type
from indibench.utils import to_pil

log = logging.getLogger(__name__)
RESOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


# ============================================================================ MMMU
@register_type("mmmu")
class MMMUTask(Task):
    """MMMU (validation): interleaved ``<image n>`` placeholders, MC and open questions."""

    default_metrics = ["accuracy"]
    needs_images_default = True

    def default_template(self) -> str:
        return "{question}"

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        opts = row.get("options")
        if isinstance(opts, str):
            try:
                opts = ast.literal_eval(opts)
            except (ValueError, SyntaxError):
                opts = []
        opts = [str(o) for o in (opts or [])]
        qtype = "mc" if (row.get("question_type") == "multiple-choice" and opts) else "open"
        images = {}
        for i in range(1, 8):
            v = row.get(f"image_{i}")
            if v is not None:
                images[i] = to_pil(v, self.root)
        q = str(row.get("question", ""))
        if qtype == "mc":
            body = q + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters(len(opts)), opts))
            body += "\n" + P.MCQ_INSTRUCTION["en"]
        else:
            body = q + "\n" + P.SHORT_ANSWER_INSTRUCTION["en"]
        ans = row.get("answer")
        if isinstance(ans, str) and ans.strip().startswith("["):
            try:
                ans = ast.literal_eval(ans)
            except (ValueError, SyntaxError):
                pass
        return Doc(id=str(row.get("id", idx)), vars={"question": body}, images=[], gold=ans,
                   choices=opts if qtype == "mc" else None, group=self.doc_group(row),
                   meta={"qtype": qtype, "image_map": images})

    def _content(self, doc: Doc) -> List[Dict[str, Any]]:
        text = self.render(doc)
        imgs = doc.meta["image_map"]
        parts: List[Dict[str, Any]] = []
        used = set()
        for chunk in re.split(r"(<image \d+>)", text):
            m = re.fullmatch(r"<image (\d+)>", chunk)
            if m:
                k = int(m.group(1))
                if k in imgs and k not in used and len(used) < self.image_cfg.max_images:
                    parts.append({"type": "image", "image": self.prepare_images([imgs[k]])[0]})
                    used.add(k)
                    continue
            if chunk:
                parts.append({"type": "text", "text": chunk})
        # Images never referenced inline are prepended.
        missing = [imgs[k] for k in sorted(imgs) if k not in used]
        lead = [{"type": "image", "image": im} for im in self.prepare_images(missing)]
        return lead + parts

    def gold_repr(self, doc: Doc) -> Any:
        return doc.gold

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        if doc.meta["qtype"] == "mc":
            idx = extract_choice(response, len(doc.choices or []), doc.choices)
            pred = None if idx is None else letters(len(doc.choices or []))[idx]
            return pred, {"accuracy": float(pred is not None and pred == str(doc.gold).strip().upper())}
        golds = doc.gold if isinstance(doc.gold, list) else [doc.gold]
        pred = response.strip().split("\n")[0]
        ok = any(_open_match(pred, str(g)) for g in golds)
        return pred, {"accuracy": float(ok)}


def _open_match(pred: str, gold: str) -> bool:
    gn, pn = M._to_float(gold), extract_number(pred)
    if gn is not None and pn is not None:
        return abs(gn - pn) <= max(1e-2 * abs(gn), 1e-6)
    g = normalize_text(gold)
    return bool(g) and (g == normalize_text(pred) or g in normalize_text(pred))


# ============================================================================ MathVista
@register_type("mathvista")
class MathVistaTask(Task):
    """MathVista testmini, rule-based answer extraction (no GPT extractor needed)."""

    default_metrics = ["accuracy"]
    needs_images_default = True

    def default_template(self) -> str:
        return "{query}"

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        img = row.get("decoded_image") or row.get("image")
        img = to_pil(img, self.root) if not isinstance(img, str) or os.path.exists(str(img)) else None
        if img is None:
            return None
        return Doc(id=str(row.get("pid", idx)), vars={"query": row.get("query") or row.get("question", "")},
                   images=[img], gold=str(row.get("answer", "")), choices=row.get("choices") or None,
                   group=self.doc_group(row),
                   meta={"qtype": row.get("question_type"), "atype": row.get("answer_type"),
                         "precision": row.get("precision")})

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        gold = doc.gold
        if doc.meta["qtype"] == "multi_choice" and doc.choices:
            idx = extract_choice(response, len(doc.choices), doc.choices)
            pred = doc.choices[idx] if idx is not None else None
            return pred, {"accuracy": float(pred is not None and normalize_text(pred) == normalize_text(gold))}
        atype = doc.meta.get("atype")
        num = extract_number(response)
        if atype == "integer":
            ok = num is not None and M._to_float(gold) is not None and int(round(num)) == int(float(gold))
            return num, {"accuracy": float(ok)}
        if atype == "float":
            prec = int(doc.meta.get("precision") or 2)
            gf = M._to_float(gold)
            ok = num is not None and gf is not None and round(num, prec) == round(gf, prec)
            return num, {"accuracy": float(ok)}
        pred = response.strip().split("\n")[-1]
        return pred, {"accuracy": float(normalize_text(gold) in normalize_text(pred))}


# ============================================================================ synthetic OCR
_SCRIPT_FONT_NAMES = {
    "Taml": ["NotoSansTamil", "NotoSerifTamil", "Lohit-Tamil", "Latha"],
    "Deva": ["NotoSansDevanagari", "NotoSerifDevanagari", "Lohit-Devanagari", "Mangal"],
    "Beng": ["NotoSansBengali", "NotoSerifBengali", "Lohit-Bengali", "Lohit-Assamese"],
    "Telu": ["NotoSansTelugu", "NotoSerifTelugu", "Lohit-Telugu"],
    "Knda": ["NotoSansKannada", "NotoSerifKannada", "Lohit-Kannada"],
    "Mlym": ["NotoSansMalayalam", "NotoSerifMalayalam", "Lohit-Malayalam"],
    "Gujr": ["NotoSansGujarati", "NotoSerifGujarati", "Lohit-Gujarati"],
    "Guru": ["NotoSansGurmukhi", "NotoSerifGurmukhi", "Lohit-Gurmukhi"],
    "Orya": ["NotoSansOriya", "NotoSerifOriya", "Lohit-Odia"],
    "Arab": ["NotoNastaliqUrdu", "NotoNaskhArabic", "NotoSansArabic"],
}
_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
              os.path.expanduser("~/.local/share/fonts"), "/Library/Fonts", "C:/Windows/Fonts",
              os.path.join(RESOURCES, "fonts")]


def find_fonts(script: str, probe_char: str) -> List[str]:
    files: List[str] = []
    for d in _FONT_DIRS:
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "**", "*.[ot]tf"), recursive=True))
    named = [f for n in _SCRIPT_FONT_NAMES.get(script, []) for f in files
             if os.path.basename(f).replace(" ", "").lower().startswith(n.lower())]
    if named:
        return sorted(set(named))
    try:  # coverage scan (optional dependency)
        from fontTools.ttLib import TTFont

        cover = []
        for f in files:
            try:
                if ord(probe_char) in TTFont(f, lazy=True, fontNumber=0).getBestCmap():
                    cover.append(f)
            except Exception:
                continue
        # Prefer real text fonts over bitmap/mono fallbacks.
        cover.sort(key=lambda f: ("unifont" in f.lower(), "mono" in f.lower(), f))
        return cover
    except ImportError:
        return []


@register_type("synthetic_ocr")
class SyntheticOCRTask(Task):
    """Render native-script text onto images and ask the model to transcribe it.

    Probes Indic-script visual reading — rarely covered by public VLM benchmarks.
    Text comes from a built-in sentence list (``text_source: builtin``) or any dataset
    column (``text_source: {dataset spec..., field: text}``). Scored by grapheme CER.
    """

    default_metrics = ["cer", "char_f1", "exact_match"]
    needs_images_default = True

    def default_template(self) -> str:
        return P.pick(P.OCR_INSTRUCTION, self.prompt_lang, self.instruction_lang)

    def load_source(self, max_rows=None) -> RowSource:
        src = self.spec.get("text_source", "builtin")
        if src == "builtin":
            path = os.path.join(RESOURCES, f"sentences_{self.lang}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"No built-in sentences for {self.lang}; set text_source to a dataset")
            with open(path, encoding="utf-8") as fh:
                rows = [{"text": ln.strip()} for ln in fh if ln.strip() and not ln.startswith("#")]
            return RowSource(rows)
        from indibench.data.loader import load_rows

        field = src.get("field", "text")
        ds = {k: v for k, v in src.items() if k != "field"}
        if self.run.trust_remote_code is not None:
            ds.setdefault("trust_remote_code", self.run.trust_remote_code)
        s = load_rows(ds, ds.get("split"), cache_dir=self.run.cache_dir, token_env=self.run.hf_token_env)
        return RowSource([{"text": t} for t in s.column(field) if t])

    def _fonts(self) -> List[str]:
        if getattr(self, "_font_cache", None) is not None:
            return self._font_cache
        if self.spec.get("font_path"):
            fonts = [self.spec["font_path"]] if isinstance(self.spec["font_path"], str) else list(self.spec["font_path"])
        else:
            lang = LANGUAGES[self.lang]
            probe = chr(lang.ranges[0][0] + 0x15)
            fonts = find_fonts(lang.script, probe)
        if not fonts:
            raise RuntimeError(
                f"No font with {self.lang} glyphs found. Install one (e.g. `apt install fonts-noto-core` "
                f"or download Noto Sans from fonts.google.com) or set font_path in the task spec.")
        from PIL import features

        if not features.check("raqm"):
            log.warning("Pillow lacks libraqm: Indic conjuncts/vowel signs will render incorrectly. "
                        "Install libraqm (apt install libraqm0) for valid OCR images.")
        self._font_cache = fonts
        return fonts

    def render_image(self, text: str, seed: int):
        from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

        rng = random.Random(seed)
        fonts = self._fonts()
        size = rng.randint(*self.spec.get("font_size_range", [30, 46]))
        font_path = fonts[rng.randrange(len(fonts))]
        engine = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
        font = ImageFont.truetype(font_path, size, layout_engine=engine)
        max_w = self.spec.get("max_width", 900)
        # greedy word wrap
        words, lines, cur = text.split(), [], ""
        probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        for w in words:
            cand = (cur + " " + w).strip()
            if probe.textlength(cand, font=font) > max_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        lines.append(cur)
        line_h = int(size * 1.6)
        width = int(max(probe.textlength(l, font=font) for l in lines)) + 60
        height = line_h * len(lines) + 50
        bg = tuple(rng.randint(225, 255) for _ in range(3))
        fg = tuple(rng.randint(0, 60) for _ in range(3))
        img = Image.new("RGB", (width, height), bg)
        d = ImageDraw.Draw(img)
        for i, l in enumerate(lines):
            d.text((30, 25 + i * line_h), l, font=font, fill=fg)
        if self.spec.get("augment", True):
            if rng.random() < 0.5:
                img = img.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=bg)
            if rng.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
        return img

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        text = str(row.get("text", "")).strip()
        max_chars = self.spec.get("max_chars", 120)
        if not text:
            return None
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0]
        img = self.render_image(text, self.run.seed * 100003 + idx)
        return Doc(id=str(idx), vars={}, images=[img], gold=[text])

    def gold_text(self, doc: Doc) -> str:
        return doc.gold[0]

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        pred = response.strip().strip('"').strip("`").strip()
        out = {}
        for m in self.metrics:
            if m in M.SAMPLE_METRICS:
                out[m] = M.SAMPLE_METRICS[m](pred, doc.gold, self.lang or "")
        return pred, out
