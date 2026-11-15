"""Exercises the transformers backend with a tiny randomly-initialised LLaVA built locally (no downloads).
Skipped when torch/transformers are not installed."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_llava(tmp_path_factory):
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


def test_hf_backend_generate_and_loglikelihood(tiny_llava, fixture_dir, tmp_path):
    from indibench.config import load_config
    from indibench.runner import Evaluator

    for mode in ("generate", "loglikelihood"):
        cfg = load_config(None, {
            "model.backend": "hf", "model.name_or_path": tiny_llava, "model.dtype": "float32",
            "model.device_map": "cpu", "run.tasks": ["local_mcq_ta", "local_vqa_xl_ta"],
            "run.task_files": [str(fixture_dir / "tasks.yaml")], "run.limit": 3, "run.batch_size": 2,
            "run.output_dir": str(tmp_path), "run.run_name": mode, "run.mcq_mode": mode,
            "generation.max_new_tokens": 4}, [])
        res = Evaluator(cfg).run()["results"]
        assert res["local_mcq_ta"]["n"] == 3 and res["local_mcq_ta"]["mode"] == mode
        assert res["local_vqa_xl_ta"]["n"] == 3
