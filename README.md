# indibench

A command-line benchmarking harness for **multimodal LLMs**, built for testing models in **Tamil and
other Indic languages**. It covers standard vision benchmarks too, and it is sized for **a single
GPU**:

* **Workstation cards:** RTX 2080 Ti (11 GB), RTX 3090 or RTX 4090 (24 GB).
* **Data-center cards:** L40S (48 GB), A100 (40/80 GB), H100 (80/94 GB) or H200 (141 GB).

* **53 task specs, expanded to 280 tasks.** They fall into five areas:
  * **Indic text:** Belebele, MILU, INCLUDE, MMLU-Indic, IndicMMLU-Pro, Global-MMLU,
    ARC/TriviaQA/BoolQ-Indic, IndicSentiment, IndicXNLI, IndicQA, IndicGenBench (XQuAD-IN,
    CrossSum-IN, Flores-IN), FLORES+, IN22-Conv, XL-Sum, MGSM, Tanglish sentiment, and Dravidian
    code-mixed offensive-language detection.
  * **Indian English:** MILU-en, MedMCQA (AIIMS/NEET-PG), JEEBench (IIT-JEE Advanced),
    TriviaQA-Indic-en, Global-MMLU-en and DRISHTIKON-en.
  * **General English:** MMLU, MMLU-Pro, GPQA-Diamond, ARC-Challenge, HellaSwag, WinoGrande,
    TruthfulQA and **IFEval**.
  * **Math and code:** GSM8K, MATH-500, AIME-24, HumanEval and MBPP.
  * **Vision:** MMMU, MathVista, ChartQA, DocVQA, TextVQA, OCRBench, AI2D, ScienceQA, MMBench,
    POPE, MME, CVQA and **DRISHTIKON** (Indian culture from images), plus Indic-script OCR and
    cross-lingual POPE probes of our own.
* **A one-page executive summary for business readers** is written after every run, and can also be
  produced from any finished run (§8a).
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

**On a fresh Ubuntu 24.04 or 26.04 machine**, follow [`docs/SETUP_UBUNTU.md`](docs/SETUP_UBUNTU.md).
It covers the NVIDIA driver (workstation and data-center/HGX, including Fabric Manager and MIG), the
optional CUDA toolkit, `uv` virtual environments, PyTorch, vLLM, Hugging Face access and
troubleshooting. You can also run the script that does all of it:

```bash
bash scripts/setup_ubuntu.sh           # driver + .venv; reboot when asked, then:
bash scripts/setup_ubuntu.sh --driver skip --vllm
```

On a machine that already has a driver ≥ 580:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision
uv pip install -e ".[hf,quant]"                       # transformers backend + bitsandbytes 4/8-bit
uv pip install flash-attn --no-build-isolation        # optional; Ampere or newer (3090/4090/L40S/A100/H100/H200), not 2080 Ti
# vLLM: put it in its own venv (it pins its own torch), see docs/SETUP_UBUNTU.md
```

Several datasets are **gated** on the Hugging Face Hub: MILU, FLORES+, Belebele and IN22. Accept their
terms on huggingface.co, then run `hf auth login` or `export HF_TOKEN=hf_xxx`.

The synthetic OCR task needs a font for each script it renders: `sudo apt install fonts-noto-core libraqm0`.

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

# Indian-English + STEM for a text model (code needs explicit opt-in)
indibench run -m Qwen/Qwen3-8B -t indian_english,math,science,code --allow-code-execution -n 200

# Text-only LLM: skip every vision task (OCR, POPE, MMMU, ...) in whatever suites you select
indibench run -m sarvamai/sarvam-m -t tamil,indic_text -l ta --text-only -n 200
indibench run -m my-llm -t full -x 'vision,xlsum_*'     # or exclude by tag / glob / group

# Compare two finished runs
indibench compare results/Qwen2.5-VL-7B-Instruct__2026* results/gemma-3-12b-it__2026*
```

The first `run` call is the important one: `-m` picks the model, `-t` picks the tasks and `-n`
caps the samples per task.

## 3. Commands

| command | purpose |
|---|---|
| `run` | Run benchmarks. Every config key has a flag, and `--set a.b=c` can override anything. `--dry-run` prints the resolved config and task list. `--text-only` / `--modality text\|image` and `-x/--exclude` narrow the task list. |
| `list [--lang ta] [--tag vision] [--modality text] [--groups]` | Browse the task catalogue. |
| `inspect TASK` | Show dataset features and rendered prompts, and save the images (`--save-images DIR`). No model needed. |
| `check TASKS` | Load a few rows of each task and parse them. Use it to catch gated or broken datasets before a long GPU run. |
| `report RUN_DIR` | Re-render `report.md` / `summary.csv`. |
| `compare RUN...` | Side-by-side primary metrics with deltas. |
| `summary RUN [--baseline RUN] [--pdf]` | One-page executive summary (HTML, PDF or Markdown) for business readers. |
| `env [-m MODEL] [--quantization 4bit]` | GPU capabilities (bf16 / FA2 / FP8), driver, auto profile and a memory-fit verdict for a model. |
| `init-config PATH` | Write an annotated example YAML. |

## 4. Configuration

Run `indibench init-config my.yaml` for a fully annotated file. The sections and their key settings:

| section | key settings |
|---|---|
| `model` | `backend`, `name_or_path`, `trust_remote_code`, `supports_images` (`false` declares a text-only model, which drops vision tasks), `dtype`, `quantization` (`4bit`/`8bit` for hf; `awq`/`gptq`/`fp8` for vllm), `attn_implementation`, `max_model_len`, `gpu_memory_utilization`, `system_prompt`, `image.max_pixels`, `image.max_images`, `api_base`, `concurrency`, `extra` (passed to `from_pretrained` / `vllm.LLM`) |
| `generation` | `max_new_tokens` (tasks may set their own), `temperature`, `top_p`, `stop`, `strip_thinking`, `enable_thinking` |
| `run` | `tasks`, `languages`, `trust_remote_code` (datasets), `modalities` (`[text]` for text-only models), `exclude`, `limit`, `sampling`, `stratify_by`, `seed`, `batch_size`, `num_fewshot`, `mcq_mode`, `instruction_lang`, `output_dir`, `run_name`, `resume`, `bootstrap`, `task_files`, `streaming`, `max_input_chars` |
| `trust_remote_code` | `true`/`false` shorthand that sets both `model.trust_remote_code` and `run.trust_remote_code` (see §9). |
| `hardware_profile` | `auto` (the default) detects the GPU. You can also give `rtx2080ti`, `rtx3090`, `rtx4090`, `l40s`, `a100-40g`, `a100-80g`, `h100`, `h200` or `none`. A profile only fills in keys you have not set yourself. |

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
| `in22conv_en-*` | conversational-domain translation (IN22; accepts `eng_Latn` or `sentence_eng_Latn` column styles) | chrF++, BLEU, script fidelity, per-domain | 16 |
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

### More Indic knowledge & generation
| family | what it measures | metrics | languages |
|---|---|---|---|
| `include_*` | INCLUDE: questions from **real regional exams written in the language** (not translated) | accuracy, per-domain | ta, te, ml, hi, bn, ur, ne |
| `mmlu_indic_*`, `arc_indic_*`, `boolq_indic_*` | MMLU / ARC-Challenge / BoolQ translated by Sarvam AI | accuracy | 10 Indic |
| `triviaqa_indic_mcq_*` | TriviaQA as 4-way MCQ (Sarvam AI) | accuracy | 10 Indic + en |
| `xquad_in_*` | IndicGenBench XQuAD-IN extractive QA | EM, F1, char-F1 | 12 |
| `crosssum_in_*` | IndicGenBench: English article → summary in the Indic language | chrF++, ROUGE-L, script fidelity | 12 |
| `flores_in_en-*` | IndicGenBench Flores-IN translation (**not gated**, unlike FLORES+) | chrF++, BLEU | 12 |
| `mgsm_*` | MGSM human-translated grade-school math | accuracy | te, bn, en |
| `tanglish_sentiment` | Tamil-English code-mixed (mostly romanised) YouTube comments | accuracy, macro-F1 | ta-en |
| `offensive_codemix_*` | offensive-content detection on code-mixed comments (moderation) | accuracy, macro-F1 | ta, ml, kn |
| `indicmmlu_pro_*` | MMLU-Pro (10 options) translated with IndicTrans2 and expert-checked | accuracy, per category | 9 incl. ta |
| `global_mmlu_*` | MMLU with a per-question culturally sensitive / agnostic label | accuracy, per label | hi, bn, te, en |
| `drishtikon_*` | Indian culture from **images**, covering all states and UTs (festivals, attire, cuisine, art) | accuracy, per attribute and state | en + Indic |

### Indian English, general English, math, science, code
| task | what it measures | scoring |
|---|---|---|
| `medmcqa` | AIIMS and NEET-PG medical entrance questions | accuracy, per subject |
| `jeebench` | IIT-JEE Advanced physics, chemistry and maths | JEEBench rules: exact for single-correct and integer answers, ±0.01 for numeric, 0.25 per option for partially correct multi-answer |
| `milu_en`, `triviaqa_indic_mcq_en` | India-centric knowledge in English | accuracy |
| `mmlu_pro`, `mmlu_en` | broad knowledge and reasoning (MMLU-Pro has 10 options) | accuracy, per category |
| `gpqa_diamond` | graduate-level science (gated); the correct answer is listed first in the data, so choices are shuffled deterministically | accuracy |
| `arc_challenge`, `hellaswag`, `winogrande`, `truthfulqa_mc1` | science reasoning, commonsense, truthfulness | accuracy |
| `gsm8k`, `math500`, `aime24` | math word problems up to olympiad level | final answer from `\boxed{}`, `####` or "The answer is", checked by normalised-LaTeX, numeric or SymPy equivalence |
| `humaneval`, `mbpp` | Python function synthesis | pass@1, by **executing unit tests** (opt-in, see below) |
| `ifeval` | following verifiable instructions ("≥ 300 words", "no commas", "answer in JSON", "end with …") | prompt- and instruction-level accuracy, strict and loose, from 25 built-in checkers with no judge model |

**Code execution is opt-in.** HumanEval and MBPP run the code the model writes. They refuse to start
unless you pass `--allow-code-execution` (or set `run.allow_code_execution: true`, or
`INDIBENCH_ALLOW_CODE_EXECUTION=1`). Each program runs in a fresh `python -I` process inside a temp
directory, with an empty environment, a timeout, and CPU, memory and file-size limits. That limits
damage but is not a security sandbox: run code benchmarks inside a container or VM.

Math and JEE tasks expect step-by-step reasoning, so their `max_new_tokens` defaults are large
(768–4096). Expect them to dominate run time, and use vLLM if you can.

### Suites and tags
* **Tamil:** `tamil`, `tamil_text`, `tamil_vision`, `tamil_open` (needs no gated access) and
  **`tamil_extended`** (everything Tamil).
* **Indic:** `indic_text`, `indic_vision`, `indic_knowledge`, `indicgenbench`.
* **English and domains:** `indian_english`, `english_core`, `math`, `science`, `code`, `stem`,
  **`indian_culture`** and **`safety`**.
* **Other:** `vision_core`, `quick` and `full`. `full` covers everything except predict-only and
  code tasks.

Any task tag can also be selected with `-t tag:<tag>`, for example `-t tag:indian_english`,
`-t tag:medical` or `-t tag:exams`. Combine suites with `--languages ta,hi`, `indic`, `indic_major`,
`dravidian` or `indo_aryan`.

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

## 7. Single-GPU guidance

indibench reads the GPU's compute capability and VRAM at start-up (`hardware_profile: auto`, the
default). It then fills in sensible defaults and makes the following automatic adjustments:

* **bf16 → fp16** on Turing (RTX 2080 Ti). PyTorch can report bf16 as "supported" there through
  slow emulation, so indibench decides from the compute capability instead.
* **FlashAttention-2 → SDPA** on cards older than Ampere.
* **Warnings** for models known to overflow in fp16 (Gemma), for FP8 on cards without FP8 tensor
  cores, for vLLM on sm_75, and when the model probably won't fit in VRAM. The warning suggests
  `8bit`/`4bit`.

| | RTX 2080 Ti | RTX 3090 | RTX 4090 | L40S | A100 40 GB | A100 80 GB | H100 80/94 GB | H200 141 GB |
|---|---|---|---|---|---|---|---|---|
| arch / CC | Turing 7.5 | Ampere 8.6 | Ada 8.9 | Ada 8.9 | Ampere 8.0 | Ampere 8.0 | Hopper 9.0 | Hopper 9.0 |
| dtype · attention | fp16 · SDPA | bf16 · FA2 | bf16 · FA2 | bf16 · FA2 | bf16 · FA2 | bf16 · FA2 | bf16 · FA2/FA3 | bf16 · FA2/FA3 |
| FP8 | – | – | ✓ | ✓ | – | – | ✓ | ✓ |
| profile | `rtx2080ti` | `rtx3090` | `rtx4090` | `l40s` | `a100-40g` | `a100-80g` | `h100` | `h200` |
| batch · context · px/image | 2 · 4k · 0.35 MP | 4 · 8k · 0.6 MP | 4 · 8k · 0.6 MP | 8 · 8k · 1 MP | 6 · 8k · 0.8 MP | 16 · 16k · 1.25 MP | 16 · 16k · 1.25 MP | 24 · 32k · 1.25 MP |
| 2–4B VLM | fp16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 |
| 7–8B VLM | 4bit | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 |
| 11–14B | – | 8bit / 4bit | fp8 / 8bit | bf16 | 8bit | bf16 | bf16 | bf16 |
| 26–38B | – | 4bit (tight) | 4bit (tight) | 4bit / AWQ | 4bit | bf16 (tight) | bf16 (tight) / fp8 | bf16 |
| 70–90B | – | – | – | – | – | 4bit / AWQ | AWQ; fp8 on NVL 94 GB | fp8 / 8bit |

MIG slices of an A100/H100/H200 are sized by the memory they expose. 8-GPU HGX (NVSwitch) systems
need Fabric Manager; see [`docs/SETUP_UBUNTU.md`](docs/SETUP_UBUNTU.md) §2b.

```bash
indibench env -m Qwen/Qwen2.5-VL-7B-Instruct      # capabilities + "will it fit?" for this GPU
indibench run -c configs/rtx2080ti_hf.yaml        # one config per GPU class: rtx3090_hf, rtx4090_vllm,
                                                  # tamil_l40s_hf, a100_80g_hf, indic_a100_vllm, h100_vllm, h200_vllm
```

* **Visual tokens drive memory use.** Every image is resized to `model.image.max_pixels`. Lower it
  (`--max-pixels 401408`) for multi-image MMMU questions or long documents.
* **vLLM is often 3–10× faster** on generation-heavy suites. On a 2080 Ti it runs with fallback
  kernels, and some architectures don't start; use `--backend hf` in that case.
* If generation runs out of memory, the hf backend splits the batch and retries. Setting
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (on by default) reduces fragmentation on
  11–24 GB cards.
* `--max-gpu-memory 10` caps VRAM use and offloads the rest of the model to CPU RAM. Larger models
  then run on small cards, at 5–20× lower speed.
* With several consumer GPUs (e.g. 2× 3090), the hf backend splits the model across them through
  `device_map=auto`. For vLLM, use `--set model.tensor_parallel_size=2`.
* `--mcq-mode loglikelihood` scores MCQ options by log-probability instead of parsing generated text.
  It works on the hf backend (text or images) and on vllm for text-only prompts.
* For reasoning models, use `--disable-thinking` or raise `--max-new-tokens`. `<think>` blocks are
  stripped before scoring.

### 7a. Keeping the GPU busy

indibench samples GPU utilisation and memory once a second for every task (via NVML or
`nvidia-smi`). The numbers go into `results.json` under `gpu`, into a **GPU utilisation** table in
`report.md`, and into a log warning with suggested fixes when a task kept the GPU busy less than
70% of the time. `--no-gpu-monitor` turns this off.

What the pipeline already does:

* **CPU work runs ahead of the GPU.** A background thread decodes dataset rows, decodes and resizes
  images and renders prompts while the model runs. `--prefetch N` (default 2) sets how many groups
  it prepares in advance; raise it to 4 for image-heavy tasks if the log reports idle gaps.
* **Large calls to engines that batch continuously.** vLLM gets 1024 text or 256 image requests
  per call, and the OpenAI-compatible client keeps `4 × --concurrency` requests in flight. Both
  ignore `--batch-size`. `--inference-chunk N` overrides the size.
* **Less padding.** Prompts are sorted by length before batching, so a batch doesn't wait on one
  very long sample.
* **Batched log-likelihood.** `--mcq-mode loglikelihood` scores all options of a batch in one
  forward pass instead of one pass per option.
* **OOM recovery is remembered.** When the hf backend halves a batch after running out of memory,
  it keeps the smaller size for the rest of the task instead of failing on every batch.
* **Code tasks score in parallel.** Unit tests for HumanEval/MBPP run on all CPU cores
  (`--set run.score_workers=N`).

What to change, by symptom:

| Symptom | Fix |
|---|---|
| hf backend, memory < 60%, GPU busy < 70% | `--batch-size auto` (starts at 64 text / 16 image and shrinks on OOM), or set a larger number |
| hf backend on generation tasks, busy in bursts | `--backend vllm`. transformers waits for the longest answer in each batch; vLLM refills slots as answers finish (typically 3–10× faster) |
| vLLM, memory looks full but cores idle | Normal: vLLM reserves `gpu_memory_utilization` (0.90) for KV cache up front. Look at the busy %, not memory |
| vLLM, busy < 70% | `--inference-chunk 2048` and `--set model.extra.max_num_seqs=512` (more sequences decoded at once); check that image decoding isn't the bottleneck (`--max-pixels`, `--prefetch 4`) |
| Image tasks, frequent idle gaps | Lower `--max-pixels`, raise `--prefetch`; store datasets on local SSD (`HF_HOME`) instead of network disks |
| OpenAI-compatible server | `--concurrency 64`–`128`; on the server, `--max-num-seqs 256` |
| Short `--limit` runs | Model load and dataset download dominate. Utilisation numbers are only meaningful for tasks running > 30 s |
| Several GPUs, one busy | hf `device_map=auto` runs layers on one GPU at a time (pipeline, not parallel). Use vLLM with `--set model.tensor_parallel_size=N`, or run one indibench process per GPU with `CUDA_VISIBLE_DEVICES` and different `--tasks` |

```bash
# typical single-GPU settings
indibench run -c configs/tamil_l40s_hf.yaml --batch-size auto --prefetch 4
indibench run -c configs/indic_a100_vllm.yaml --inference-chunk 2048
watch -n1 nvidia-smi        # or: nvtop
```

## 8a. Executive summary for business readers

Every run writes `executive_summary.html` to its run folder. This is a single A4 page meant for
people who will never open `results.json`. You can also build one from any finished run, optionally
comparing it against another run:

```bash
indibench summary results/<run>                                # HTML in the run folder
indibench summary results/<run> --baseline results/<other_run> --pdf
indibench summary results/<run> --format md                    # paste into email / Confluence
indibench summary results/<run> --narrative-model <model> --narrative-api-base http://localhost:8000/v1
```

The page contains:

* **Headline KPIs:** overall score, Tamil score, Tamil-vs-English parity on the same tests, and how
  much evidence the scores rest on (number of tests, questions and the typical margin of error).
* **A capability scorecard** with nine business areas: Tamil understanding, Tamil writing &
  translation, other Indian languages, Indian context, English, math & science, programming,
  vision & documents, and safety & reliability. Each area gets a 0–100 score, a rating (Strong,
  Good, Limited or Weak) and a plain-language verdict ("Ready for production use", "Usable with
  human review", "Pilot only", "Not recommended yet"). With `--baseline`, a Δ column shows the
  change against the other run.
* **Strengths, watch-outs and recommendations.** These are generated by rules from the numbers. For
  example: Tamil text that keeps drifting into English, a Tamil-vs-English gap, areas not tested
  in this run, or sample sizes too small for a final decision.

How the scores are calculated:

* **Skill scores** rescale multiple-choice accuracy so that random guessing scores 0 and a perfect
  result scores 100. Without this, a 25% score on a 4-option test would look like partial success.
* Error rates such as OCR CER are inverted. Other measures (chrF++, pass@1) are used as they are.
* The page contains no AI-written text unless you pass `--narrative-model`. That option sends only
  the computed scores to the model, and the resulting paragraph is labelled as AI-drafted.

`--pdf` uses WeasyPrint or a headless Chrome/Chromium if one is installed. Otherwise, open the HTML
and use Print → Save as PDF. Turn the automatic summary off with `--set run.executive_summary=false`.
[`docs/sample_executive_summary.pdf`](docs/sample_executive_summary.pdf) shows the layout, using
synthetic scores.

## 8. Outputs

```
results/<model>__<timestamp>/
  config.yaml            fully resolved config (reproducibility)
  env.json               GPU, CUDA, library versions
  predictions/<task>.jsonl   per-sample prompt, raw response, parsed prediction, gold, metrics
  results.json           all metrics + CIs + breakdowns + timing + peak GPU memory
  summary.csv            one row per task
  report.md              tables, language matrix, per-group breakdowns
  executive_summary.html one-page business summary (indibench summary … --pdf for a PDF)
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

A field can name alternative columns: `source: "eng_Latn|sentence_eng_Latn"` uses the first column
that exists. Mirrors of the same dataset often rename columns, and this handles that.

The `dataset` block accepts `path`, `name`/`names` (several configs are concatenated), `data_dir`,
`data_files` (including `hf://datasets/...` URLs and local json/jsonl/parquet/csv), `split`, `filter`,
`preprocess` (`squad_flatten`, `drop_unanswerable`, `literal_eval`, `require_image`), `streaming`,
`trust_remote_code`, `parquet_fallback`
and `pair` (which zips two configs of a parallel corpus). The language placeholders are `{iso}`,
`{name}`, `{Name}`, `{native}`, `{flores}` and `{xlsum}`.

### Text-only models
Every task is tagged `text` or `image` according to whether it sends images to the model
(`indibench list` shows a modality column). There are three ways to run a text-only LLM:

* **`--text-only`** (same as `--modality text`, or `run.modalities: [text]` in YAML) drops every
  vision task from the suites you selected.
* **`--set model.supports_images=false`** (or `model.supports_images: false`) declares the model
  text-only. indibench then drops vision tasks automatically.
* **Automatic detection.** The hf and vllm backends detect text-only checkpoints themselves and skip
  vision tasks with a warning. The openai backend can't tell, so use one of the two options above
  for API models.

`-x/--exclude` removes any tasks by name, glob, tag or group, for example `-x vision`,
`-x 'xlsum_*,mmlu_en'` or `-x vision_core`.

### Remote code (`trust_remote_code`)
Remote code is **off by default** for both models and datasets. You can turn it on or off per
component. Only enable it for repositories you trust: it runs Python code downloaded from the Hub.

| what | YAML | CLI |
|---|---|---|
| model + processor (custom architectures such as InternVL, MiniCPM-V, Phi-3.5-vision) | `model.trust_remote_code: true` | `--model-trust-remote-code [true\|false]` |
| dataset loading scripts | `run.trust_remote_code: true`, or per task `dataset.trust_remote_code` | `--dataset-trust-remote-code [true\|false]` |
| both at once | `trust_remote_code: true` (top level) | `--trust-remote-code [true\|false]` (bare flag = true) |

The more specific key wins, so `trust_remote_code: true` with `run.trust_remote_code: false` trusts
only the model.

For **datasets**, this setting only takes effect with `datasets` < 4, because version 4.0 removed
the ability to run loading scripts. With newer `datasets`, indibench logs a warning and reads the
Hub's auto-converted Parquet copy instead (below). To run a dataset's original script, use an
environment with `pip install "datasets<4"`.

### How datasets are loaded, and fixing one
The loader copes with two common Hub problems automatically:

* **Legacy loading scripts.** `datasets` ≥ 4 refuses repositories that ship a `*.py` loader, such as
  XL-Sum. When this happens, indibench reads the Hub's auto-converted Parquet copy on the
  `refs/convert/parquet` branch. It handles both file layouts used there
  (`<config>/<prefix>-<split>.parquet` and `<config>/<split>/NNNN.parquet`) and logs a warning.
  Set `dataset.parquet_fallback: false` to turn this off.
* **Files in unknown folders.** `dataset.hub_files: {repo: org/name, pattern: "*english-ta_test.json"}`
  lists the repository's files and loads whatever matches, whether it is at the root or in a
  subfolder. IndicGenBench uses this.
* **One table for all languages.** `filter: {"language|lang": [ta, Tamil]}` filters on the first of
  those columns that exists. `config_fallback_filter` switches to such a filter when a per-language
  config doesn't exist. A filter on a missing column fails with a list of the real columns. BoolQ-Indic
  uses this.
* **"Container" JSON files.** Some files hold one JSON object whose single key wraps all the rows
  (IndicXNLI), or are column-oriented (`{"col": [...], ...}`). These are unwrapped into normal rows.
  Previously this showed up as *No row produced a valid sample*.

If `indibench check X` still fails:

1. Run `indibench inspect X` to see the features the loader actually returns.
2. Copy the spec from `indibench/tasks/specs/*.yaml` into your own task file under the **same name**
   (yours overrides the built-in one) and adjust `dataset.data_files` or `fields`.
3. Run `indibench check X` again.

## 10. Notes on the built-in specs

Each spec has a `status:` line. MILU, IndicQA, IndicSentiment, IndicXNLI, IN22-Conv and CVQA column
names were checked against their dataset cards. The other benchmarks use their widely used Hub
schemas. The IndicXNLI raw-file layout (`forward/test/xnli_<iso>.json`, a single-key JSON object)
was checked against the repository and its original loading script. XL-Sum is read from the Hub's
auto-converted Parquet. The IndicSentiment raw-file path is inferred from the repository layout, so
run `indibench check indic_text -l ta` once on your machine. Scores from this harness use
its own prompts and parsers. Compare models *within* indibench rather than against numbers from
papers.

## 11. Development

```bash
pip install -e ".[dev]" sympy
pytest -q        # offline tests: metrics, sampling, resume, GPU capability logic, dummy/openai/hf backends
```
