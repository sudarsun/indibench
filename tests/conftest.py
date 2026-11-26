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


@pytest.fixture(scope="session")
def tiny_llava(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (CLIPImageProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration, LlavaProcessor, PreTrainedTokenizerFast)

    corpus = ["Answer with the option's letter A B C D yes no", "இன்று வானிலை சென்னை தமிழ்நாட்டின்", "user assistant system"]*50
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>","<s>","</s>","<pad>","<image>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders
    tok.decoder = decoders.ByteLevel()
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    ft.add_special_tokens({"additional_special_tokens": ["<image>"]})
    tmpl = ("{% for m in messages %}<|{{ m['role'] }}|>"
            "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}"
            "{% if c['type'] == 'image' %}<image>{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    ft.chat_template = tmpl
    V = len(ft)
    text_cfg = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024, pad_token_id=ft.pad_token_id, bos_token_id=ft.bos_token_id, eos_token_id=ft.eos_token_id)
    vis_cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8, projection_dim=32)
    cfg = LlavaConfig(vision_config=vis_cfg, text_config=text_cfg, image_token_index=ft.convert_tokens_to_ids("<image>"), vision_feature_layer=-1)
    m = LlavaForConditionalGeneration(cfg)
    ip = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    proc = LlavaProcessor(image_processor=ip, tokenizer=ft, patch_size=8, vision_feature_select_strategy="default", chat_template=tmpl, num_additional_image_tokens=1)
    
    out = tmp_path_factory.mktemp("tiny-llava")
    m.save_pretrained(out)
    proc.save_pretrained(out)
    return str(out)


