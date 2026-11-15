"""Language metadata used to expand task templates and score language fidelity.

Each language exposes several *code conventions* because every benchmark uses a
different one (ISO-639-1 ``ta``, FLORES ``tam_Taml``, XL-Sum ``tamil``, MILU ``Tamil`` ...).
Task specs reference these through ``{iso}``, ``{flores}``, ``{name}``, ``{xlsum}``, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Language:
    iso: str                       # ISO 639-1 (or 639-3 when no 639-1 exists)
    name: str                      # English name, e.g. "Tamil"
    native: str                    # Endonym, e.g. "தமிழ்"
    flores: str                    # FLORES-200 / IndicTrans2 code, e.g. "tam_Taml"
    script: str                    # ISO 15924 script code
    ranges: Tuple[Tuple[int, int], ...]  # Unicode ranges of the primary script
    xlsum: Optional[str] = None    # csebuetnlp/xlsum config name
    yes: Tuple[str, ...] = ()      # words meaning "yes"
    no: Tuple[str, ...] = ()       # words meaning "no"
    extra: Dict[str, str] = field(default_factory=dict)

    def vars(self) -> Dict[str, str]:
        """Variables available to task-spec templates."""
        v = {
            "lang": self.iso,
            "iso": self.iso,
            "name": self.name,
            "Name": self.name,
            "name_lower": self.name.lower(),
            "native": self.native,
            "flores": self.flores,
            "script": self.script,
            "xlsum": self.xlsum or self.name.lower(),
            "ISO": self.iso.upper(),
            "yes_word": self.yes[0] if self.yes else "yes",
            "no_word": self.no[0] if self.no else "no",
        }
        v.update(self.extra)
        return v


TAMIL = (0x0B80, 0x0BFF)
DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)
GURMUKHI = (0x0A00, 0x0A7F)
GUJARATI = (0x0A80, 0x0AFF)
ORIYA = (0x0B00, 0x0B7F)
TELUGU = (0x0C00, 0x0C7F)
KANNADA = (0x0C80, 0x0CFF)
MALAYALAM = (0x0D00, 0x0D7F)
ARABIC = (0x0600, 0x06FF)
ARABIC_EXT = (0x0750, 0x077F)
OL_CHIKI = (0x1C50, 0x1C7F)
MEETEI = (0xABC0, 0xABFF)
LATIN = (0x0041, 0x024F)

LANGUAGES: Dict[str, Language] = {
    l.iso: l
    for l in [
        Language("ta", "Tamil", "தமிழ்", "tam_Taml", "Taml", (TAMIL,), "tamil",
                 ("ஆம்", "ஆமாம்", "உண்டு"), ("இல்லை", "கிடையாது")),
        Language("hi", "Hindi", "हिन्दी", "hin_Deva", "Deva", (DEVANAGARI,), "hindi",
                 ("हाँ", "हां", "जी हाँ"), ("नहीं", "ना")),
        Language("te", "Telugu", "తెలుగు", "tel_Telu", "Telu", (TELUGU,), "telugu",
                 ("అవును",), ("కాదు", "లేదు")),
        Language("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda", "Knda", (KANNADA,), None,
                 ("ಹೌದು",), ("ಇಲ್ಲ", "ಅಲ್ಲ")),
        Language("ml", "Malayalam", "മലയാളം", "mal_Mlym", "Mlym", (MALAYALAM,), None,
                 ("അതെ", "ഉണ്ട്"), ("ഇല്ല", "അല്ല")),
        Language("bn", "Bengali", "বাংলা", "ben_Beng", "Beng", (BENGALI,), "bengali",
                 ("হ্যাঁ", "হাঁ"), ("না",)),
        Language("mr", "Marathi", "मराठी", "mar_Deva", "Deva", (DEVANAGARI,), "marathi",
                 ("हो", "होय"), ("नाही",)),
        Language("gu", "Gujarati", "ગુજરાતી", "guj_Gujr", "Gujr", (GUJARATI,), "gujarati",
                 ("હા",), ("ના", "નહીં")),
        Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "pan_Guru", "Guru", (GURMUKHI,), "punjabi",
                 ("ਹਾਂ",), ("ਨਹੀਂ",)),
        Language("or", "Odia", "ଓଡ଼ିଆ", "ory_Orya", "Orya", (ORIYA,), None,
                 ("ହଁ",), ("ନା", "ନାହିଁ")),
        Language("as", "Assamese", "অসমীয়া", "asm_Beng", "Beng", (BENGALI,), None,
                 ("হয়",), ("নহয়", "নাই")),
        Language("ur", "Urdu", "اردو", "urd_Arab", "Arab", (ARABIC, ARABIC_EXT), "urdu",
                 ("ہاں", "جی ہاں"), ("نہیں",)),
        Language("ne", "Nepali", "नेपाली", "npi_Deva", "Deva", (DEVANAGARI,), "nepali",
                 ("हो", "हजुर"), ("होइन", "छैन")),
        Language("sa", "Sanskrit", "संस्कृतम्", "san_Deva", "Deva", (DEVANAGARI,), None),
        Language("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "sat_Olck", "Olck", (OL_CHIKI,), None),
        Language("mni", "Manipuri", "ꯃꯤꯇꯩꯂꯣꯟ", "mni_Mtei", "Mtei", (MEETEI,), None),
        Language("en", "English", "English", "eng_Latn", "Latn", (LATIN,), "english",
                 ("yes",), ("no",)),
    ]
}

# Convenience groupings for CLI filters: ``--languages indic`` or ``--languages dravidian``
LANGUAGE_GROUPS: Dict[str, List[str]] = {
    "dravidian": ["ta", "te", "kn", "ml"],
    "indo_aryan": ["hi", "bn", "mr", "gu", "pa", "or", "as", "ur", "ne"],
    "indic": ["ta", "te", "kn", "ml", "hi", "bn", "mr", "gu", "pa", "or", "as", "ur"],
    "indic_major": ["ta", "hi", "te", "kn", "ml", "bn", "mr", "gu"],
}


def get_language(code: str) -> Language:
    code = code.strip()
    if code in LANGUAGES:
        return LANGUAGES[code]
    for lang in LANGUAGES.values():
        if code.lower() in (lang.flores.lower(), lang.name.lower(), (lang.xlsum or "").lower()):
            return lang
    raise KeyError(f"Unknown language code {code!r}. Known: {sorted(LANGUAGES)}")


def expand_language_filter(codes: Optional[List[str]]) -> Optional[List[str]]:
    """Expand group aliases (``indic``, ``dravidian``) into ISO codes."""
    if not codes:
        return None
    out: List[str] = []
    for c in codes:
        c = c.strip()
        if not c:
            continue
        if c in LANGUAGE_GROUPS:
            out.extend(LANGUAGE_GROUPS[c])
        else:
            out.append(get_language(c).iso)
    return list(dict.fromkeys(out))


def script_ratio(text: str, lang: Language) -> float:
    """Fraction of *letter-like* characters in ``text`` that belong to ``lang``'s script.

    Digits, punctuation and whitespace are ignored so that "42" or "A." do not
    penalise a response. Returns 0.0 for empty/no-letter strings.
    """
    total = 0
    hit = 0
    for ch in text:
        if ch.isspace() or ch.isdigit() or not (ch.isalpha() or _is_combining_mark(ch)):
            continue
        total += 1
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in lang.ranges):
            hit += 1
    return hit / total if total else 0.0


def _is_combining_mark(ch: str) -> bool:
    import unicodedata

    return unicodedata.category(ch) in ("Mn", "Mc")
