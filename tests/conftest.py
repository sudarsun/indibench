"""Builds tiny local datasets (Tamil text + images) so the full pipeline runs offline."""
import json
import os

import pytest

TA_MCQ = [
    {"q": "தமிழ்நாட்டின் தலைநகரம் எது?", "opts": ["மதுரை", "சென்னை", "கோவை", "சேலம்"], "a": "option2", "dom": "geo"},
    {"q": "ஒரு வாரத்தில் எத்தனை நாட்கள்?", "opts": ["ஐந்து", "ஆறு", "ஏழு", "எட்டு"], "a": "option3", "dom": "math"},
    {"q": "திருக்குறளை எழுதியவர் யார்?", "opts": ["கம்பர்", "இளங்கோ", "பாரதியார்", "திருவள்ளுவர்"], "a": "option4", "dom": "lit"},
    {"q": "காவிரி ஒரு ...", "opts": ["ஆறு", "மலை", "கடல்", "ஏரி"], "a": "option1", "dom": "geo"},
    {"q": "2 + 3 = ?", "opts": ["4", "5", "6", "7"], "a": "option2", "dom": "math"},
    {"q": "சிலப்பதிகாரம் எழுதியவர்?", "opts": ["இளங்கோவடிகள்", "கம்பர்", "ஔவையார்", "பாரதிதாசன்"], "a": "option1", "dom": "lit"},
]


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    from PIL import Image, ImageDraw

    d = tmp_path_factory.mktemp("data")
    with open(d / "mcq_ta.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(TA_MCQ):
            fh.write(json.dumps({"id": i, "question": r["q"], "option1": r["opts"][0], "option2": r["opts"][1],
                                 "option3": r["opts"][2], "option4": r["opts"][3], "target": r["a"],
                                 "domain": r["dom"]}, ensure_ascii=False) + "\n")
    with open(d / "qa_ta.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "q1", "context": "சென்னை தமிழ்நாட்டின் தலைநகரம் ஆகும். இது கடற்கரை நகரம்.",
                             "question": "தமிழ்நாட்டின் தலைநகரம் எது?", "answers": ["சென்னை"]}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"id": "q2", "context": "மதுரையில் மீனாட்சி அம்மன் கோயில் உள்ளது.",
                             "question": "மீனாட்சி அம்மன் கோயில் எங்கே உள்ளது?", "answers": ["மதுரையில்", "மதுரை"]},
                            ensure_ascii=False) + "\n")
    with open(d / "mt_en_ta.jsonl", "w", encoding="utf-8") as fh:
        for en, ta in [("The weather is pleasant today.", "இன்று வானிலை இனிமையாக உள்ளது."),
                       ("Children are playing in the park.", "குழந்தைகள் பூங்காவில் விளையாடுகிறார்கள்."),
                       ("Chennai is the capital of Tamil Nadu.", "சென்னை தமிழ்நாட்டின் தலைநகரம்.")]:
            fh.write(json.dumps({"en": en, "ta": ta}, ensure_ascii=False) + "\n")
    os.makedirs(d / "imgs", exist_ok=True)
    rows = []
    for i, (color, ans) in enumerate([("red", "yes"), ("blue", "no"), ("red", "yes"), ("green", "no")]):
        img = Image.new("RGB", (64, 48), color)
        ImageDraw.Draw(img).rectangle([10, 10, 30, 30], outline="white")
        img.save(d / "imgs" / f"{i}.png")
        rows.append({"qid": i, "image": f"imgs/{i}.png", "question": "Is the image red?", "answer": ans,
                     "category": color})
    with open(d / "vqa.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    spec = f"""
tasks:
  - name: local_mcq_{{lang}}
    type: mcq
    languages: [ta]
    dataset: {{path: "{d}/mcq_ta.jsonl", split: test}}
    group_by: domain
    fields: {{id: id, question: question, choices: [option1, option2, option3, option4], answer: target,
              answer_format: option_key}}
  - name: local_qa_ta
    type: qa
    lang: ta
    dataset: {{path: "{d}/qa_ta.jsonl", split: test}}
    fields: {{id: id, context: context, question: question, answers: answers}}
  - name: local_mt_en-ta
    type: generation
    lang: ta
    source_lang: en
    target_lang: ta
    dataset: {{path: "{d}/mt_en_ta.jsonl", split: test}}
    fields: {{source: en, target: ta}}
    metrics: [chrf, bleu, script_fidelity]
  - name: local_vqa_xl_ta
    type: vqa
    lang: ta
    target_lang: ta
    dataset: {{path: "{d}/vqa.jsonl", split: test}}
    group_by: category
    fields: {{id: qid, image: image, question: question, answers: answer}}
    metrics: [yesno_accuracy, script_fidelity]
groups:
  local_all: [local_mcq_ta, local_qa_ta, "local_mt_*", local_vqa_xl_ta]
"""
    (d / "tasks.yaml").write_text(spec, encoding="utf-8")
    return d
