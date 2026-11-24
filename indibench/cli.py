"""indibench command-line interface.

    indibench run      --model Qwen/Qwen2.5-VL-7B-Instruct --tasks tamil --limit 200
    indibench run      --config configs/tamil_l40s.yaml --set run.limit=50
    indibench list     [--lang ta] [--tag vision] [--groups]
    indibench inspect  belebele_ta            # schema + rendered prompts, no model needed
    indibench check    tamil                   # can every task load & parse? (fast)
    indibench report   results/<run>
    indibench compare  results/run_a results/run_b
    indibench env                              # GPU info + recommended settings
    indibench init-config my.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from indibench import __version__

log = logging.getLogger("indibench")


def _limit(v: str):
    if v.lower() in ("all", "none", "0"):
        return "all"
    try:
        return int(v)
    except ValueError:
        f = float(v)
        if not 0 < f <= 1:
            raise argparse.ArgumentTypeError("--limit float must be a fraction in (0, 1]")
        return f


def _csv(v: str) -> List[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="indibench", description="Benchmark (multimodal) LLMs on Tamil, Indic and vision tasks.")
    p.add_argument("--version", action="version", version=f"indibench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run benchmarks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("-c", "--config", help="YAML config file (CLI flags override it)")
    g = r.add_argument_group("model")
    g.add_argument("-m", "--model", dest="model.name_or_path", help="HF model id / local path / API model name")
    g.add_argument("-b", "--backend", dest="model.backend", choices=["hf", "vllm", "openai", "dummy"])
    g.add_argument("--dtype", dest="model.dtype")
    g.add_argument("--quantization", dest="model.quantization", help="4bit|8bit (hf, bitsandbytes) or awq|gptq|fp8 (vllm)")
    g.add_argument("--attn", dest="model.attn_implementation", help="sdpa|flash_attention_2|eager")
    g.add_argument("--trust-remote-code", dest="model.trust_remote_code", action="store_const", const=True)
    g.add_argument("--max-model-len", dest="model.max_model_len", type=int)
    g.add_argument("--gpu-mem-util", dest="model.gpu_memory_utilization", type=float, help="vLLM memory fraction")
    g.add_argument("--max-pixels", dest="model.image.max_pixels", type=int, help="per-image pixel budget")
    g.add_argument("--max-images", dest="model.image.max_images", type=int)
    g.add_argument("--api-base", dest="model.api_base", help="OpenAI-compatible endpoint (backend=openai)")
    g.add_argument("--concurrency", dest="model.concurrency", type=int)
    g.add_argument("--system-prompt", dest="model.system_prompt")
    g.add_argument("--hardware-profile", dest="hardware_profile",
                   help="auto (default) | none | rtx2080ti | rtx3090 | rtx4090 | l40s | a100-40g | a100-80g | h100 | h200")
    g.add_argument("--max-gpu-memory", dest="model.max_gpu_memory_gb", type=float,
                   help="hf: cap GiB per GPU and offload the rest to CPU RAM")
    g = r.add_argument_group("generation")
    g.add_argument("--max-new-tokens", dest="generation.max_new_tokens", type=int)
    g.add_argument("--temperature", dest="generation.temperature", type=float)
    g.add_argument("--enable-thinking", dest="generation.enable_thinking", action="store_const", const=True)
    g.add_argument("--disable-thinking", dest="generation.enable_thinking", action="store_const", const=False)
    g = r.add_argument_group("run")
    g.add_argument("-t", "--tasks", dest="run.tasks", type=_csv, help="comma list of tasks, globs or groups")
    g.add_argument("-l", "--languages", dest="run.languages", type=_csv, help="e.g. ta,hi or indic / dravidian")
    g.add_argument("-n", "--limit", dest="run.limit", type=_limit,
                   help="samples per task: N, a fraction like 0.1, or 'all'")
    g.add_argument("--sampling", dest="run.sampling", choices=["random", "first", "stratified"])
    g.add_argument("--stratify-by", dest="run.stratify_by")
    g.add_argument("--seed", dest="run.seed", type=int)
    g.add_argument("--batch-size", dest="run.batch_size", type=int)
    g.add_argument("--num-fewshot", dest="run.num_fewshot", type=int)
    g.add_argument("--mcq-mode", dest="run.mcq_mode", choices=["generate", "loglikelihood"])
    g.add_argument("--instruction-lang", dest="run.instruction_lang", choices=["en", "native"])
    g.add_argument("-o", "--output-dir", dest="run.output_dir")
    g.add_argument("--run-name", dest="run.run_name", help="reuse a name to resume an interrupted run")
    g.add_argument("--no-resume", dest="run.resume", action="store_const", const=False)
    g.add_argument("--streaming", dest="run.streaming", action="store_const", const=True)
    g.add_argument("--bootstrap", dest="run.bootstrap", type=int)
    g.add_argument("--task-file", dest="run.task_files", action="append", help="extra task-spec YAML (repeatable)")
    g.add_argument("--log-level", dest="run.log_level")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key, e.g. --set model.extra.mode=oracle")
    r.add_argument("--dry-run", action="store_true", help="resolve config + tasks, print, and exit")

    ls = sub.add_parser("list", help="list tasks and groups")
    ls.add_argument("--lang", type=_csv)
    ls.add_argument("--tag", type=_csv)
    ls.add_argument("--groups", action="store_true", help="show groups only")
    ls.add_argument("--task-file", action="append", default=[])
    ls.add_argument("--json", action="store_true")

    ins = sub.add_parser("inspect", help="show dataset schema and rendered prompts for a task (no model)")
    ins.add_argument("task")
    ins.add_argument("-n", type=int, default=2)
    ins.add_argument("--instruction-lang", choices=["en", "native"], default="en")
    ins.add_argument("--num-fewshot", type=int)
    ins.add_argument("--task-file", action="append", default=[])
    ins.add_argument("--save-images", help="directory to dump the images of the shown samples")

    ck = sub.add_parser("check", help="verify that tasks load and parse (downloads data, no model)")
    ck.add_argument("tasks", type=_csv)
    ck.add_argument("--languages", type=_csv)
    ck.add_argument("--task-file", action="append", default=[])
    ck.add_argument("--rows", type=int, default=5)

    rp = sub.add_parser("report", help="re-render reports for a finished run")
    rp.add_argument("run_dir")

    cp = sub.add_parser("compare", help="compare several runs")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("-o", "--output", help="write markdown to file")

    for a in r._actions:
        if a.dest and "." in a.dest and a.metavar is None and a.nargs != 0 and a.const is None:
            a.metavar = a.dest.split(".")[-1].upper()

    ev = sub.add_parser("env", help="show GPU / library info, capability checks and model-fit advice")
    ev.add_argument("-m", "--model", help="model id to check memory fit for, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    ev.add_argument("--quantization", help="4bit | 8bit | awq | gptq | fp8")
    ev.add_argument("--dtype", default="bfloat16")

    ic = sub.add_parser("init-config", help="write an annotated example config")
    ic.add_argument("path")
    return p


# --------------------------------------------------------------------------- commands
def cmd_run(args) -> int:
    from indibench.config import load_config
    from indibench.report import print_summary
    from indibench.runner import Evaluator
    from indibench.utils import setup_logging

    cli_values: Dict[str, Any] = {k: v for k, v in vars(args).items() if "." in k or k == "hardware_profile"}
    if cli_values.get("run.limit") == "all":
        cli_values["run.limit"] = None
        args.overrides = list(args.overrides) + ["run.limit=null"]
    cfg = load_config(args.config, cli_values, args.overrides)
    setup_logging(cfg.run.log_level)
    if not cfg.run.tasks:
        log.error("No tasks given (use --tasks or run.tasks in the YAML). See `indibench list`.")
        return 2
    if args.dry_run:
        from indibench.tasks.registry import TaskRegistry
        from indibench.languages import expand_language_filter
        import yaml

        reg = TaskRegistry(cfg.run.task_files)
        names = reg.resolve(cfg.run.tasks, expand_language_filter(cfg.run.languages))
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
        print(f"# {len(names)} tasks: {', '.join(names)}")
        return 0
    ev = Evaluator(cfg)
    payload = ev.run()
    print_summary(payload)
    print(f"\nResults written to {ev.run_dir}/ (report.md, summary.csv, results.json, predictions/)")
    errors = [k for k, v in payload["results"].items() if "error" in v]
    return 1 if errors and len(errors) == len(payload["results"]) else 0


def cmd_list(args) -> int:
    from indibench.languages import expand_language_filter
    from indibench.tasks.registry import TaskRegistry

    reg = TaskRegistry(args.task_file)
    if args.groups:
        for g, members in reg.groups.items():
            desc = reg.group_desc.get(g, "")
            try:
                n = len(reg.resolve([g]))
            except KeyError:
                n = "?"
            print(f"{g:16s} {str(n):>4s} tasks  {desc}")
        return 0
    langs = expand_language_filter(args.lang) if args.lang else None
    rows = []
    for name, s in sorted(reg.specs.items()):
        if langs and s.get("lang") not in langs:
            continue
        if args.tag and not set(args.tag) & set(s.get("tags") or []):
            continue
        rows.append((name, s.get("type"), s.get("lang") or "-", ",".join(s.get("tags") or []),
                     (s.get("description") or "").strip()))
    if args.json:
        print(json.dumps([dict(zip(["name", "type", "lang", "tags", "description"], r)) for r in rows],
                         ensure_ascii=False, indent=2))
        return 0
    try:
        from rich.console import Console
        from rich.table import Table

        t = Table(show_lines=False)
        for c in ("task", "type", "lang", "tags", "description"):
            t.add_column(c, overflow="fold")
        for r in rows:
            t.add_row(r[0], r[1], r[2], r[3], r[4][:90])
        Console().print(t)
    except ImportError:
        for r in rows:
            print("\t".join(r))
    print(f"{len(rows)} tasks. Groups: {', '.join(reg.groups)}")
    return 0


def _make_task(name: str, task_files, instruction_lang="en", num_fewshot=None):
    from indibench.config import ImageConfig, RunConfig
    from indibench.tasks.base import build_task
    from indibench.tasks.registry import TaskRegistry

    reg = TaskRegistry(task_files)
    run = RunConfig(instruction_lang=instruction_lang, num_fewshot=num_fewshot)
    return build_task(reg.get(name), run, ImageConfig())


def cmd_inspect(args) -> int:
    from indibench.utils import setup_logging

    setup_logging("WARNING")
    task = _make_task(args.task, args.task_file, args.instruction_lang, args.num_fewshot)
    print(f"== {task.name}  (type={task.type_name}, lang={task.lang}, modality={task.modality})")
    print(f"   dataset: {json.dumps(task.dataset_spec(), ensure_ascii=False)}")
    src = task.load_source(max_rows=args.n + 20)
    print(f"   rows: {len(src)}")
    print("   features:")
    for k, v in src.features().items():
        print(f"     - {k}: {v[:110]}")
    shown = 0
    for i, row in enumerate(src.rows(range(min(len(src), args.n * 5)))):
        doc = task.make_doc(row, i)
        if doc is None:
            continue
        req = task.build_request(doc)
        print(f"\n--- sample id={doc.id} images={len(req.images)} gold={task.gold_repr(doc)!r}")
        for m in req.messages:
            text = "\n".join(p["text"] for p in m["content"] if p["type"] == "text")
            nimg = sum(1 for p in m["content"] if p["type"] == "image")
            print(f"[{m['role']}{' +' + str(nimg) + ' img' if nimg else ''}]\n{text}")
        if args.save_images:
            os.makedirs(args.save_images, exist_ok=True)
            for j, im in enumerate(req.images):
                path = os.path.join(args.save_images, f"{task.name}_{doc.id.replace('/', '_')}_{j}.png")
                im.save(path)
                print(f"(saved {path})")
        shown += 1
        if shown >= args.n:
            break
    if shown == 0:
        print("\n!! No row produced a valid sample — fix the `fields` mapping for this task.")
        return 1
    return 0


def cmd_check(args) -> int:
    from indibench.languages import expand_language_filter
    from indibench.tasks.registry import TaskRegistry
    from indibench.utils import setup_logging

    setup_logging("ERROR")
    reg = TaskRegistry(args.task_file)
    names = reg.resolve(args.tasks, expand_language_filter(args.languages))
    ok = 0
    for name in names:
        try:
            task = _make_task(name, args.task_file)
            src = task.load_source(max_rows=args.rows * 4)
            docs = [task.make_doc(r, i) for i, r in enumerate(src.rows(range(min(len(src), args.rows * 4))))]
            good = [d for d in docs if d is not None]
            if not good:
                raise RuntimeError(f"0/{len(docs)} rows parsed — field mapping mismatch")
            task.build_request(good[0])
            print(f"OK    {name:28s} rows={len(src):>7}  parsed={len(good)}/{len(docs)}")
            ok += 1
        except Exception as e:
            print(f"FAIL  {name:28s} {type(e).__name__}: {str(e)[:220]}")
    print(f"\n{ok}/{len(names)} tasks OK")
    return 0 if ok == len(names) else 1


def cmd_report(args) -> int:
    from indibench.report import print_summary, write_reports

    with open(os.path.join(args.run_dir, "results.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    write_reports(payload, args.run_dir)
    print_summary(payload)
    return 0


def cmd_compare(args) -> int:
    from indibench.report import compare_runs

    md = compare_runs(args.runs)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    return 0


def cmd_env(args) -> int:
    from indibench.config import HARDWARE_PROFILES
    from indibench.hardware import check_fit, detect_gpus, profile_for
    from indibench.runner import environment_info

    info = environment_info()
    print(json.dumps(info, indent=2))
    gpus = detect_gpus()
    if not gpus:
        print("\nNo CUDA GPU visible to PyTorch. Check `nvidia-smi`, the driver (>= 580 for CUDA 13 wheels) "
              "and that torch was installed with CUDA (torch.version.cuda above should not be null).")
        return 1
    for i, g in enumerate(gpus):
        yes = lambda b: "yes" if b else "NO"  # noqa: E731
        print(f"\nGPU{i}: {g.name} | {g.arch} sm_{g.capability[0]}{g.capability[1]} | {g.memory_gb:.1f} GB | "
              f"bf16 {yes(g.bf16)} | FlashAttention-2 {yes(g.flash_attn2)} | FP8 {yes(g.fp8)}")
        if not g.supported_by_torch_cu13:
            print("  !! Pre-Turing GPU: current PyTorch CUDA-13 wheels have no kernels for it.")
    prof = profile_for(gpus[0])
    print(f"\nAuto hardware profile: {prof}")
    print("  fills unset keys:", json.dumps(HARDWARE_PROFILES[prof]))
    if args.model:
        dtype = args.dtype if gpus[0].bf16 or args.dtype == "float32" else "float16"
        fit = check_fit(args.model, dtype, args.quantization, gpus[0], len(gpus))
        print(f"\nModel fit for {args.model} ({dtype}, quantization={args.quantization or 'none'}):")
        if fit.verdict == "unknown":
            print("  could not infer the parameter count from the name")
        else:
            print(f"  ~{fit.params_b:g}B params -> ~{fit.weights_gb:.1f} GB weights on {fit.gpu_gb:.0f} GB "
                  f"total VRAM: {fit.verdict.upper()}{' - ' + fit.advice if fit.advice else ''}")
    print("""
Sizing guide (weights only; add 2-8 GB for activations, KV cache and visual tokens):
            2080Ti 11G  3090/4090 24G  L40S 48G   A100 40G   A100 80G / H100 80G  H200 141G
  2-4B VLM  fp16        bf16           bf16       bf16       bf16                 bf16
  7-8B VLM  4bit        bf16           bf16       bf16       bf16                 bf16
  11-14B    --          8bit/4bit/fp8* bf16       8bit       bf16                 bf16
  26-38B    --          4bit (tight)   4bit/AWQ   4bit       bf16 (tight)/8bit    bf16
  70-90B    --          --             --         --         4bit/AWQ (H100: fp8*) 8bit/fp8*
  * fp8 = vLLM --quantization fp8, Ada/Hopper only (RTX 4090, L40S, H100, H200).
Turing (2080 Ti): float16 only, no FlashAttention-2, no FP8. Use --backend hf if vLLM fails to start.""")
    return 0


EXAMPLE_CONFIG = """\
# indibench example configuration — every key can be overridden on the CLI with --set key=value
hardware_profile: auto          # auto | none | rtx2080ti | rtx3090 | rtx4090 | l40s | a100-40g | a100-80g | h100 | h200

model:
  backend: hf                   # hf | vllm | openai | dummy
  name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
  dtype: bfloat16
  attn_implementation: sdpa     # flash_attention_2 if installed
  quantization: null            # 4bit | 8bit (hf) ; awq | gptq | fp8 (vllm)
  trust_remote_code: false
  max_model_len: 8192
  gpu_memory_utilization: 0.90  # vllm
  system_prompt: null
  image:
    max_pixels: 1003520         # ~1 MP per image keeps visual tokens bounded
    max_images: 8
  # api_base: http://localhost:8000/v1   # for backend: openai (vllm serve / sglang / ollama)

generation:
  max_new_tokens: 256
  temperature: 0.0
  strip_thinking: true

run:
  tasks: [tamil]                # tasks, globs (belebele_*) or groups (tamil, indic_text, vision_core, quick)
  languages: null               # e.g. [ta, hi] or [indic] to filter language-expanded tasks
  limit: 200                    # per task: int count, fraction (0.1), or null for all
  sampling: stratified          # random | first | stratified (uses task group_by, e.g. MILU subject)
  seed: 1234
  batch_size: 8
  num_fewshot: null             # override per-task default (MILU/MMLU support few-shot)
  mcq_mode: null                # generate | loglikelihood (hf/vllm text) — null = per-task default
  instruction_lang: en          # en | native (Tamil/Hindi instructions where available)
  output_dir: results
  resume: true
  bootstrap: 1000
  task_files: []                # your own task YAMLs
"""


def cmd_init_config(args) -> int:
    if os.path.exists(args.path):
        print(f"{args.path} exists; not overwriting", file=sys.stderr)
        return 1
    with open(args.path, "w", encoding="utf-8") as fh:
        fh.write(EXAMPLE_CONFIG)
    print(f"Wrote {args.path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # Reduce CUDA memory fragmentation — matters most on 11-24 GB cards. Must be set before torch init.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_run, "list": cmd_list, "inspect": cmd_inspect, "check": cmd_check,
                "report": cmd_report, "compare": cmd_compare, "env": cmd_env, "init-config": cmd_init_config}
    try:
        code = handlers[args.cmd](args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    
    #sys.exit(code)
    return code


if __name__ == "__main__":
    main()
