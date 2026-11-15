# indibench

A command-line benchmarking harness for **multimodal LLMs**, built for testing models in **Tamil and
other Indic languages**. It covers standard vision benchmarks too, and it is sized for **one L40S or
A100 GPU**.

* **25 task specs, expanded to 149 tasks.** Belebele, MILU, IndicSentiment, IndicXNLI, IndicQA, FLORES+,
  IN22-Conv, XL-Sum, MMMU, MathVista, ChartQA, DocVQA, TextVQA, OCRBench, AI2D, ScienceQA, MMBench,
  POPE, MME, CVQA. There are also two Indic vision probes of our own: synthetic Indic-script OCR and
  a cross-lingual POPE.
* **You choose how many samples to run.** Use `--limit 200`, `--limit 0.1` or `--limit all`. Sampling
  is deterministic (seeded) and can be *stratified* so that small runs still cover every MILU subject.
* **Four backends.** `hf` (transformers, with optional 4/8-bit), `vllm` (highest throughput),
  `openai` (any OpenAI-compatible server) and `dummy` (offline pipeline checks).
* **Configuration from the CLI, YAML, or both.** Precedence is defaults < YAML < CLI flags <
  `--set key=value`.
* **Indic-aware scoring.** Normalisation handles danda, ZWJ and native digits. Metrics include
  character-F1 (needed for agglutinative languages), grapheme-cluster CER and script fidelity
  (did the model actually answer in Tamil?). Prompts can be given in English or in the native
  language (`--instruction-lang native`).
* **Built for long runs.** Predictions are written to disk as the run goes, so you can resume. On
  CUDA OOM the batch is split and retried. Every metric gets a bootstrap 95% CI. Results come out as
  per-group breakdowns, a family × language matrix, Markdown/CSV/JSON reports and run-to-run
  comparisons.

---

## 1. Install

```bash
cd indibench
pip install -e ".[hf]"            # transformers backend (torch, transformers, accelerate)
pip install -e ".[hf,quant]"      # + bitsandbytes 4/8-bit
pip install -e ".[vllm]"          # + vLLM backend
pip install flash-attn --no-build-isolation   # optional; works on L40S (Ada) and A100 (Ampere)
```

Several datasets are **gated** on the Hugging Face Hub: MILU, FLORES+, Belebele and IN22. Accept their
terms on huggingface.co, then run:

```bash
export HF_TOKEN=hf_xxx
```

The synthetic OCR task needs a font for each script it renders. On Debian/Ubuntu:
`sudo apt install fonts-noto-core libraqm0`. libraqm is needed for correct Indic glyph shaping.
Pillow wheels usually include it already.

## 2. Quick start

```bash
indibench env                                   # GPU, library versions, sizing advice
indibench list --groups                         # suites
indibench list --lang ta                        # every Tamil task
indibench check tamil                           # can each task download + parse? (seconds, no model)
indibench inspect milu_ta -n 2 --instruction-lang native   # schema + exact prompts

# Tamil suite, 200 samples per task, on an L40S
indibench run -m Qwen/Qwen2.5-VL-7B-Instruct -t tamil -n 200 --hardware-profile l40s

# All Indic text tasks for the Dravidian languages via vLLM, few-shot MILU, stratified sampling
indibench run -b vllm -m google/gemma-3-12b-it -t indic_text -l dravidian -n 300 \
              --sampling stratified --num-fewshot 5 --batch-size 64

# From YAML, overriding a few keys
indibench run -c configs/tamil_l40s_hf.yaml -n 50 --set model.quantization=4bit

# Compare two finished runs
indibench compare results/Qwen2.5-VL-7B-Instruct__2026* results/gemma-3-12b-it__2026*
```

The first `run` call is the important one: `-m` picks the model, `-t` picks the tasks and `-n`
caps the samples per task.

## 3. Commands

| command | purpose |
|---|---|
| `run` | Run benchmarks. Every config key has a flag, and `--set a.b=c` can override anything. `--dry-run` prints the resolved config and task list. |
| `list [--lang ta] [--tag vision] [--groups]` | Browse the task catalogue. |
| `inspect TASK` | Show dataset features and rendered prompts, and save the images (`--save-images DIR`). No model needed. |
| `check TASKS` | Load a few rows of each task and parse them. Use it to catch gated or broken datasets before a long GPU run. |
| `report RUN_DIR` | Re-render `report.md` / `summary.csv`. |
| `compare RUN...` | Side-by-side primary metrics with deltas. |
| `env` | GPU and library versions, the detected hardware profile and model-size guidance. |
| `init-config PATH` | Write an annotated example YAML. |

## 4. Configuration

Run `indibench init-config my.yaml` for a fully annotated file. The sections and their key settings:

| section | key settings |
|---|---|
| `model` | `backend`, `name_or_path`, `dtype`, `quantization` (`4bit`/`8bit` for hf; `awq`/`gptq`/`fp8` for vllm), `attn_implementation`, `max_model_len`, `gpu_memory_utilization`, `system_prompt`, `image.max_pixels`, `image.max_images`, `api_base`, `concurrency`, `extra` (passed to `from_pretrained` / `vllm.LLM`) |
| `generation` | `max_new_tokens` (tasks may set their own), `temperature`, `top_p`, `stop`, `strip_thinking`, `enable_thinking` |
| `run` | `tasks`, `languages`, `limit`, `sampling`, `stratify_by`, `seed`, `batch_size`, `num_fewshot`, `mcq_mode`, `instruction_lang`, `output_dir`, `run_name`, `resume`, `bootstrap`, `task_files`, `streaming`, `max_input_chars` |
| `hardware_profile` | `l40s`, `a100-40g`, `a100-80g` or `auto`. A profile only fills in keys you have not set yourself. |

**Sample selection.** `limit` sets the number of samples per task. It can be an integer, a fraction,
or `null` for all. `sampling: random` takes a seeded random subset. `first` takes the first N rows.
`stratified` takes an equal share from each value of the task's `group_by` column (or `--stratify-by
col`). Re-running with a larger `--limit` and the same `--run-name` reuses the predictions already
made and only runs the new samples.

## 5. Task catalogue

`indibench list` shows the live catalogue. Language-expanded tasks are named `<family>_<iso>`, for
example `belebele_ta` and `flores_en-ta`.

### Indic text
| family | what it measures | metrics | languages |
|---|---|---|---|
| `belebele_*` | reading comprehension (FLORES passages, 4-way MCQ) | accuracy | 13 Indic + en |
| `milu_*` | India-centric knowledge across 8 domains and 41 subjects (AI4Bharat) | accuracy, per-domain/subject breakdown | 10 Indic + en |
| `indicsentiment_*` | sentiment classification | accuracy, macro-F1 | 12 |
| `indicxnli_*` | natural language inference | accuracy, macro-F1 | 11 |
| `indicqa_*` | extractive QA over Indic Wikipedia | EM, token-F1, **char-F1** | 11 |
| `flores_en-*`, `flores_*-en` | translation in both directions | chrF++, BLEU (flores200 spm), script fidelity | 13 |
| `in22conv_en-*` | conversational-domain translation (IN22) | chrF++, BLEU, script fidelity, per-domain | 16 |
| `xlsum_*` | abstractive summarisation | Unicode-aware ROUGE-L, chrF++, script fidelity | 9 |
| `mmlu_en` | English knowledge anchor | accuracy | en |

### Indic vision
| task | what it measures |
|---|---|
| `synth_ocr_ta`, `synth_ocr_hi` | Reading Tamil/Hindi script in images. Sentences and signage are rendered with random fonts, sizes, blur and rotation, and scored by **grapheme-cluster CER**. No dataset download is needed. |
| `synth_ocr_flores_*` | The same probe for 11 scripts, using FLORES+ sentences. |
| `pope_xl_*` | Cross-lingual hallucination probe. The POPE object question stays in English but the answer must be the yes/no word in the target language, which tests grounding, instruction following and output-language control together. |
| `cvqa_*` | Culturally grounded VQA written by native speakers (Tamil-India and others). **Predict-only**: the test labels are hidden, so predictions are saved in `predictions/cvqa_ta.jsonl` for leaderboard submission. |

### Standard multimodal (`vision_core`)
`mmmu_val` (images interleaved at their `<image n>` positions), `mathvista_testmini` (rule-based
extraction), `chartqa` (relaxed accuracy), `docvqa_val` (ANLS), `textvqa_val` (VQA soft accuracy),
`ocrbench`, `ai2d`, `scienceqa_img`, `mmbench_en_dev`, `pope`, `mme`.

### Suites
`tamil`, `tamil_text`, `tamil_vision`, `tamil_open` (needs no gated access), `indic_text`,
`indic_vision`, `vision_core`, `quick`, `full`. Combine a suite with `--languages ta,hi`, `indic`,
`indic_major`, `dravidian` or `indo_aryan`.

## 6. Why the Indic-specific metrics matter

* **Character-F1.** Tamil is agglutinative. The answer `சென்னையில்` ("in Chennai") gets token-F1 = 0
  against `சென்னை`, which is unfairly harsh. Char-F1 gives it 0.8. The report shows both.
* **Grapheme CER.** A Tamil "letter" such as `மி` is a consonant plus a vowel sign, i.e. two code
  points. CER is computed over grapheme clusters, so a missing vowel sign counts as one error, not two.
* **Script fidelity.** This is the share of the response's letters written in the target script.
  Many models drift into English or romanised Tamil. This metric catches that drift even when the
  content is right.
* **Normalisation.** Before matching, text is NFC-normalised, danda and other Unicode punctuation are
  removed, ZWJ/ZWNJ are stripped, and Tamil/Devanagari digits are mapped to ASCII.
* **Native instructions.** `--instruction-lang native` switches instructions and labels
  (வினா / விடை / பத்தி …) to Tamil or Hindi. Comparing `en` with `native` shows whether the model
  depends on English scaffolding. The extractors also read answers written as `விடை: B`, `ஆம்`,
  `இல்லை` and so on.

## 7. Single-GPU guidance (L40S 48 GB / A100 40–80 GB)

| model size | L40S 48 GB | A100 40 GB | A100 80 GB |
|---|---|---|---|
| ≤ 8B | bf16, batch 8–16 | bf16, batch 6–8 | bf16, batch 16+ |
| 11–14B | bf16, batch 4–8 | `--quantization 8bit` or 4bit | bf16 |
| 26–38B | `--quantization 4bit` (hf) or an AWQ checkpoint (vllm) | 4bit, tight | bf16 (tight) or 8bit |
| 70B+ | not recommended | not recommended | AWQ/GPTQ 4-bit with vLLM |

* **Visual tokens drive memory use.** Every image is resized to `model.image.max_pixels` (about 1 MP
  by default). Lower it with `--max-pixels 602112` for multi-image MMMU questions or long documents.
* **vLLM is often 3–10× faster** on generation-heavy suites such as translation, summarisation and
  OCR. Use `--batch-size 64` or more; vLLM does the scheduling itself.
* `--mcq-mode loglikelihood` scores MCQ options by log-probability instead of parsing generated text.
  It works on the hf backend with or without images, and on the vllm backend for text-only prompts.
* If generation runs out of memory, the hf backend splits the batch and retries automatically.
* For reasoning models, use `--disable-thinking` or raise `--max-new-tokens`. `<think>` blocks are
  stripped before scoring.

## 8. Outputs

```
results/<model>__<timestamp>/
  config.yaml            fully resolved config (reproducibility)
  env.json               GPU, CUDA, library versions
  predictions/<task>.jsonl   per-sample prompt, raw response, parsed prediction, gold, metrics
  results.json           all metrics + CIs + breakdowns + timing + peak GPU memory
  summary.csv            one row per task
  report.md              tables, language matrix, per-group breakdowns
```

## 9. Adding your own tasks

Write a YAML file and pass it with `--task-file`. See `examples/custom_tasks.yaml`, which runs on
the bundled local Tamil VQA sample:

```yaml
tasks:
  - name: mybench_{lang}            # expanded once per language
    type: mcq                       # mcq | qa | vqa | generation | mmmu | mathvista | synthetic_ocr
    languages: [ta, te, kn, ml]
    dataset: {path: my-org/my-bench, name: "{flores}", split: test}   # or path: ./data.jsonl
    fields: {question: q, choices: [a, b, c, d], answer: label, answer_format: letter}
    group_by: subject
    metrics: [accuracy]
    prompt:
      template: "{L_question}: {question}\n{choices_block}\nAnswer with the letter."
      template_native: {ta: "வினா: {question}\n{choices_block}\nசரியான எழுத்தை மட்டும் எழுதவும்."}
```

The `dataset` block accepts `path`, `name`/`names` (several configs are concatenated), `data_dir`,
`data_files` (including `hf://datasets/...` URLs and local json/jsonl/parquet/csv), `split`, `filter`,
`preprocess` (`squad_flatten`, `drop_unanswerable`, `literal_eval`, `require_image`), `streaming`
and `pair` (which zips two configs of a parallel corpus). The language placeholders are `{iso}`,
`{name}`, `{Name}`, `{native}`, `{flores}` and `{xlsum}`.

### Fixing a dataset
Hub datasets sometimes move files or drop their loading scripts. Recent versions of `datasets`
refuse script-based repositories. If `indibench check X` fails:

1. Run `indibench inspect X` to see the features the loader actually returns.
2. Copy the spec from `indibench/tasks/specs/*.yaml` into your own task file under the **same name**
   (yours overrides the built-in one) and adjust `dataset.data_files` or `fields`.
3. Run `indibench check X` again.

## 10. Notes on the built-in specs

Each spec has a `status:` line. MILU, IndicQA, IndicSentiment, IndicXNLI, IN22-Conv and CVQA column
names were checked against their dataset cards. The other benchmarks use their widely used Hub
schemas. The raw-file paths for IndicXNLI and IndicSentiment were inferred from the repository
layout, so run `indibench check indic_text -l ta` once on your machine. Scores from this harness use
its own prompts and parsers. Compare models *within* indibench rather than against numbers from
papers.

## 11. Development

```bash
pip install -e ".[dev]"
pytest -q        # offline tests: metrics, sampling, resume, dummy/openai/hf backends (tiny local LLaVA)
```
