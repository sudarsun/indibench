"""Indic-aware text normalisation used by every string-matching metric.

Standard SQuAD normalisation (lowercase, strip ASCII punctuation, drop English articles)
silently breaks for Indic scripts: it leaves danda (।), zero-width joiners and native
digits in place and may split on combining marks. This module handles those cases.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_EN_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)


def _native_digits_to_ascii(text: str) -> str:
    out = []
    for ch in text:
        if ch.isdigit() and not ("0" <= ch <= "9"):
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


def normalize_text(text: str, lang: str = "", remove_articles: bool = True, keep_zwj: bool = False) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    if not keep_zwj:
        text = text.translate(_ZERO_WIDTH)
    text = _native_digits_to_ascii(text)
    text = text.lower()
    # Remove all Unicode punctuation & symbols (includes । ॥ and fullwidth punctuation).
    text = "".join(" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text)
    if remove_articles and lang in ("", "en"):
        text = _EN_ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str, lang: str = "") -> List[str]:
    return normalize_text(text, lang).split()


def graphemes(text: str) -> List[str]:
    """Approximate extended grapheme clusters: base char + following combining marks/virama.

    Good enough for CER on Indic scripts, where a 'character' as perceived by a reader is
    a consonant plus vowel sign(s)/virama, not a single code point.
    """
    clusters: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if clusters and (cat in ("Mn", "Mc", "Me") or ch in "‌‍"):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters
