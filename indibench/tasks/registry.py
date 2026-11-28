"""Load task specs from YAML, expand per-language templates, resolve names/globs/groups."""
from __future__ import annotations

import copy
import fnmatch
import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from indibench.languages import LANGUAGES
from indibench.utils import fmt

log = logging.getLogger(__name__)
SPEC_DIR = os.path.join(os.path.dirname(__file__), "specs")

# Keys formatted with language variables at expansion time. ``prompt`` is deliberately
# excluded: its placeholders are filled per-document at render time.
_EXPAND_KEYS = ("name", "dataset", "fields", "tags", "source_lang", "target_lang", "choices",
                "text_source", "vars", "description", "answer_map", "fewshot_split", "family")


_IMAGE_TYPES = {"mmmu", "mathvista", "synthetic_ocr"}


def spec_modality(spec: Dict[str, Any]) -> str:
    """``image`` if the task feeds images to the model, else ``text`` (decided from the spec alone)."""
    if spec.get("modality"):
        return spec["modality"]
    f = spec.get("fields") or {}
    if spec.get("type") in _IMAGE_TYPES or f.get("image") or f.get("images"):
        return "image"
    return "text"


class TaskRegistry:
    def __init__(self, extra_files: Optional[List[str]] = None):
        self.specs: Dict[str, Dict[str, Any]] = {}
        self.groups: Dict[str, List[str]] = {}
        self.group_desc: Dict[str, str] = {}
        # optional per-group filters: {"modality": "text"|"image", "exclude": [...]}
        self.group_filters: Dict[str, Dict[str, Any]] = {}
        files = sorted(glob.glob(os.path.join(SPEC_DIR, "*.yaml")))
        for f in files + list(extra_files or []):
            self.load_file(f)

    def load_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for raw in data.get("tasks", []):
            for spec in expand_spec(raw):
                spec["_source"] = os.path.basename(path)
                spec["modality"] = spec_modality(spec)
                if spec["name"] in self.specs and self.specs[spec["name"]]["_source"] != spec["_source"]:
                    log.info("Task %s from %s overrides built-in", spec["name"], path)
                self.specs[spec["name"]] = spec
        for gname, g in (data.get("groups") or {}).items():
            if isinstance(g, dict):
                self.groups[gname] = list(g.get("tasks", []))
                self.group_desc[gname] = g.get("description", "")
                filt = {k: g[k] for k in ("modality", "exclude") if g.get(k)}
                if filt:
                    self.group_filters[gname] = filt
            else:
                self.groups[gname] = list(g)

    def resolve(self, patterns: List[str], languages: Optional[List[str]] = None,
                _seen: Optional[set] = None, modalities: Optional[List[str]] = None,
                exclude: Optional[List[str]] = None) -> List[str]:
        """Expand task names, globs (``belebele_*``) and groups into an ordered list.

        ``modalities`` keeps only tasks of those modalities (``text`` / ``image``);
        ``exclude`` drops tasks matching any of the given names, globs, tags or groups.
        """
        seen = _seen if _seen is not None else set()
        out: List[str] = []
        for p in patterns:
            p = p.strip()
            if not p:
                continue
            if p in self.groups:
                if p in seen:
                    continue
                seen.add(p)
                filt = self.group_filters.get(p, {})
                mod = filt.get("modality")
                excl = filt.get("exclude")
                out.extend(self.resolve(self.groups[p], languages, seen,
                                        modalities=[mod] if isinstance(mod, str) else mod,
                                        exclude=[excl] if isinstance(excl, str) else excl))
                continue
            if p.startswith("tag:"):           # tag:math, tag:indian_english ...
                tag = p[4:]
                matches = sorted(n for n, sp in self.specs.items() if tag in (sp.get("tags") or []))
                if not matches:
                    raise KeyError(f"No task has tag {tag!r}. Try `indibench list --tag ...`.")
                out.extend(matches)
                continue
            matches = [n for n in self.specs if fnmatch.fnmatchcase(n, p)]
            if not matches:
                raise KeyError(f"No task or group matches {p!r}. Try `indibench list`.")
            out.extend(sorted(matches) if any(c in p for c in "*?[") else matches)
        out = list(dict.fromkeys(out))
        if languages:
            out = [n for n in out if self.specs[n].get("lang") in (None, *languages)]
        if modalities and "all" not in modalities:
            out = [n for n in out if self.specs[n].get("modality", "text") in modalities]
        if exclude:
            out = [n for n in out if not self._excluded(n, exclude)]
        return out

    def _excluded(self, name: str, patterns: List[str]) -> bool:
        tags = set(self.specs[name].get("tags") or [])
        for p in patterns:
            p = p.strip()
            if not p:
                continue
            if fnmatch.fnmatchcase(name, p) or p in tags:
                return True
            if p in self.groups and name in self.resolve([p]):
                return True
        return False

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self.specs:
            raise KeyError(f"Unknown task {name!r}")
        return copy.deepcopy(self.specs[name])


def expand_spec(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A spec with ``languages: [ta, hi, ...]`` becomes one task per language."""
    langs = raw.get("languages")
    family = raw.get("family") or raw["name"].replace("_{lang}", "").replace("{lang}", "").replace("-{lang}", "")
    family = family.replace("{iso}", "").strip("_-") or raw["name"]
    if not langs:
        s = copy.deepcopy(raw)
        s.setdefault("family", family)
        return [s]
    out = []
    for code in langs:
        if code not in LANGUAGES:
            raise KeyError(f"Spec {raw['name']}: unknown language {code!r}")
        lv = LANGUAGES[code].vars()
        overrides = (raw.get("per_language") or {}).get(code, {})
        s = copy.deepcopy(raw)
        s.pop("languages", None)
        s.pop("per_language", None)
        for k in _EXPAND_KEYS:
            if k in s:
                s[k] = fmt(s[k], lv)
        s = _merge(s, overrides)
        s["lang"] = code
        s["family"] = family
        out.append(s)
    return out


def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
