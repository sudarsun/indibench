"""Default prompt fragments, with native-language variants for ``run.instruction_lang: native``.

Only languages listed here get native instructions; others fall back to English. Add a
language by adding entries to each dict (or override templates per task in YAML).
"""
from __future__ import annotations

from typing import Dict

LABELS: Dict[str, Dict[str, str]] = {
    "en": {"passage": "Passage", "question": "Question", "answer": "Answer", "hint": "Hint",
           "options": "Options", "premise": "Premise", "hypothesis": "Hypothesis", "text": "Text"},
    "ta": {"passage": "பத்தி", "question": "வினா", "answer": "விடை", "hint": "குறிப்பு",
           "options": "விருப்பங்கள்", "premise": "முன்கூற்று", "hypothesis": "கருதுகோள்", "text": "உரை"},
    "hi": {"passage": "अनुच्छेद", "question": "प्रश्न", "answer": "उत्तर", "hint": "संकेत",
           "options": "विकल्प", "premise": "आधार वाक्य", "hypothesis": "परिकल्पना", "text": "पाठ"},
}

MCQ_INSTRUCTION = {
    "en": "Answer with the option's letter from the given choices directly.",
    "ta": "கொடுக்கப்பட்ட விருப்பங்களிலிருந்து சரியான விடையின் எழுத்தை மட்டும் பதிலாக அளிக்கவும்.",
    "hi": "दिए गए विकल्पों में से सही उत्तर का केवल अक्षर लिखें।",
}

EXTRACTIVE_QA_INSTRUCTION = {
    "en": "Read the passage and answer the question with a short span copied exactly from the passage. "
          "Output only the answer.",
    "ta": "பின்வரும் பத்தியைப் படித்து, வினாவிற்கான விடையைப் பத்தியிலிருந்து நேரடியாக ஒரு சிறிய "
          "சொற்றொடராக மட்டும் அளிக்கவும்.",
    "hi": "निम्नलिखित अनुच्छेद पढ़ें और प्रश्न का उत्तर अनुच्छेद से सीधे एक छोटे वाक्यांश के रूप में दें। "
          "केवल उत्तर लिखें।",
}

SHORT_ANSWER_INSTRUCTION = {
    "en": "Answer the question using a single word or phrase.",
    "ta": "ஒரு சொல் அல்லது சிறிய சொற்றொடரில் மட்டும் விடையளிக்கவும்.",
    "hi": "प्रश्न का उत्तर एक शब्द या छोटे वाक्यांश में दें।",
}

YESNO_INSTRUCTION = {
    "en": "Answer with yes or no only.",
    "ta": "ஆம் அல்லது இல்லை என்று மட்டும் பதிலளிக்கவும்.",
    "hi": "केवल हाँ या नहीं में उत्तर दें।",
}

SUMMARY_INSTRUCTION = {
    "en": "Summarize the following {name} article in one or two sentences, written in {name}.",
    "ta": "பின்வரும் கட்டுரையைத் தமிழில் ஓரிரு வாக்கியங்களில் சுருக்கமாகக் கூறவும்.",
    "hi": "निम्नलिखित लेख का हिंदी में एक या दो वाक्यों में सारांश लिखें।",
}

OCR_INSTRUCTION = {
    "en": "Transcribe all the text in the image exactly as written. Output only the text.",
    "ta": "படத்தில் உள்ள உரையை எழுதியுள்ளபடியே சரியாக எழுதவும். உரையை மட்டும் அளிக்கவும்.",
    "hi": "चित्र में लिखे पूरे पाठ को ठीक वैसे ही लिखें जैसा लिखा है। केवल पाठ लिखें।",
}

TRANSLATION_INSTRUCTION = {
    "en": "Translate the following {src_name} text into {tgt_name}. Output only the {tgt_name} translation.",
}


def pick(table: Dict[str, str], lang: str, instruction_lang: str) -> str:
    if instruction_lang == "native" and lang in table:
        return table[lang]
    return table["en"]


def labels(lang: str, instruction_lang: str) -> Dict[str, str]:
    if instruction_lang == "native" and lang in LABELS:
        return LABELS[lang]
    return LABELS["en"]
