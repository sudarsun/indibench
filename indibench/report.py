"""Console / Markdown / CSV reports, language matrices and run comparison."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from indibench.utils import fmt_duration


def _fmt(v: Optional[float], digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{digits}f}"


def _primary(res: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    pm = res.get("primary_metric")
    m = (res.get("metrics") or {}).get(pm) if pm else None
    return pm, m


def _bench_seconds(payload: Dict[str, Any]) -> float:
    return sum(((r.get("timing") or {}).get("seconds") or 0) for r in payload.get("results", {}).values())


def _total_line(payload: Dict[str, Any]) -> str:
    total = payload.get("total_s") or payload.get("elapsed_s") or 0
    parts = []
    if payload.get("model_load_s") is not None:
        parts.append(f"model load {fmt_duration(payload['model_load_s'])}")
    parts.append(f"benchmarks {fmt_duration(_bench_seconds(payload))}")
    return f"total time: {fmt_duration(total)} ({', '.join(parts)})"


def task_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, res in payload.get("results", {}).items():
        row: Dict[str, Any] = {"task": name, "family": res.get("family"), "lang": res.get("lang") or ""}
        if "error" in res or "skipped" in res:
            row["status"] = "error" if "error" in res else "skipped"
            row["note"] = (res.get("error") or res.get("skipped") or "")[:160]
            row["seconds"] = (res.get("timing") or {}).get("seconds")
            rows.append(row)
            continue
        pm, m = _primary(res)
        row.update(status="predict_only" if res.get("predict_only") else "ok", n=res.get("n"),
                   metric=pm, value=m["value"] if m else None,
                   ci_low=m.get("ci_low") if m else None, ci_high=m.get("ci_high") if m else None,
                   others={k: v["value"] for k, v in (res.get("metrics") or {}).items() if k != pm},
                   seconds=(res.get("timing") or {}).get("seconds"),
                   peak_mem=res.get("peak_gpu_mem_gb"))
        rows.append(row)
    return rows


def language_matrix(payload: Dict[str, Any]) -> Tuple[List[str], "OrderedDict[str, Dict[str, float]]", Dict[str, str]]:
    """family × language table of primary-metric values (only language-expanded tasks)."""
    langs: List[str] = []
    mat: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    metric_of: Dict[str, str] = {}
    for name, res in payload.get("results", {}).items():
        lang = res.get("lang")
        if not lang or "metrics" not in res or res.get("predict_only"):
            continue
        pm, m = _primary(res)
        if not m:
            continue
        fam = res.get("family") or name
        mat.setdefault(fam, {})[lang] = m["value"]
        metric_of[fam] = pm or ""
        if lang not in langs:
            langs.append(lang)
    # Tamil first, English last, rest in first-seen order.
    order = list(langs)
    langs = sorted(langs, key=lambda l: (l != "ta", l == "en", order.index(l)))
    return langs, mat, metric_of


def render_markdown(payload: Dict[str, Any]) -> str:
    model = payload.get("model", {})
    env = payload.get("env", {})
    cfg = payload.get("config", {})
    lines = [f"# indibench report — `{model.get('name', '?')}`", ""]
    gpu = env.get("gpu", {})
    lines.append(f"- backend: **{model.get('backend', '?')}**, params: {model.get('params_b', '?')}B, "
                 f"dtype: {model.get('dtype', cfg.get('model', {}).get('dtype'))}, "
                 f"quantization: {model.get('quantization') or 'none'}")
    if gpu:
        lines.append(f"- GPU: {gpu.get('name')} ({gpu.get('memory_gb')} GB, CC {gpu.get('compute_capability', '?')}, "
                     f"driver {gpu.get('driver', '?')}, CUDA {gpu.get('cuda', '?')})")
    run = cfg.get("run", {})
    lines.append(f"- limit per task: {run.get('limit') or 'all'}, sampling: {run.get('sampling')}, seed: {run.get('seed')}, "
                 f"instruction_lang: {run.get('instruction_lang')}")
    lines.append(f"- {_total_line(payload)}")
    for note in payload.get("hardware_notes") or []:
        lines.append(f"- ⚠ hardware: {note}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| task | lang | n | metric | score | 95% CI | other metrics | time |")
    lines.append("|---|---|---:|---|---:|---|---|---:|")
    for r in task_rows(payload):
        if r["status"] in ("error", "skipped"):
            lines.append(f"| {r['task']} | {r['lang']} | – | {r['status']} | – | – | {r['note'].replace('|', '/')} | "
                         f"{fmt_duration(r.get('seconds')) if r.get('seconds') is not None else '–'} |")
            continue
        ci = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]" if r.get("ci_low") is not None else ""
        others = ", ".join(f"{k}={_fmt(v)}" for k, v in (r.get("others") or {}).items())
        score = "predict-only" if r["status"] == "predict_only" else _fmt(r.get("value"))
        lines.append(f"| {r['task']} | {r['lang']} | {r.get('n')} | {r.get('metric') or ''} | {score} | {ci} | "
                     f"{others} | {fmt_duration(r.get('seconds'))} |")
    lines.append(f"| **total** | | | | | | | **{fmt_duration(_bench_seconds(payload))}** |")
    langs, mat, metric_of = language_matrix(payload)
    if mat and (len(langs) > 1 or any(len(v) > 1 for v in mat.values())):
        lines += ["", "## Language matrix (primary metric)", ""]
        lines.append("| family | metric | " + " | ".join(langs) + " |")
        lines.append("|---|---|" + "---:|" * len(langs))
        for fam, vals in mat.items():
            lines.append(f"| {fam} | {metric_of.get(fam, '')} | " + " | ".join(_fmt(vals.get(l)) for l in langs) + " |")
    # breakdowns
    bd_lines = []
    for name, res in payload.get("results", {}).items():
        for key, groups in (res.get("breakdown") or {}).items():
            if len(groups) < 2:
                continue
            bd_lines.append(f"### {name} by `{key}` ({res.get('primary_metric')})")
            bd_lines.append("")
            bd_lines.append("| group | n | score |")
            bd_lines.append("|---|---:|---:|")
            for g, v in groups.items():
                bd_lines.append(f"| {g} | {v['n']} | {_fmt(v['value'])} |")
            bd_lines.append("")
    if bd_lines:
        lines += ["", "## Breakdowns", ""] + bd_lines
    perf = [(n, r) for n, r in payload.get("results", {}).items() if r.get("gpu")]
    if any(r.get("gpu") for _, r in perf):
        lines += ["", "## GPU utilisation", "", "| task | batch | samples/s | GPU busy % | idle % | peak mem % |",
                  "|---|---:|---:|---:|---:|---:|"]
        tips = []
        for n, r in perf:
            g = r.get("gpu") or {}
            sps = (r.get("timing") or {}).get("samples_per_s")
            lines.append(f"| {n} | {r.get('batch_size', '')} | {_fmt(sps)} | {_fmt(g.get('util_avg'))} | "
                         f"{_fmt(g.get('idle_share'))} | {_fmt(g.get('mem_peak_pct'))} |")
            tips += [t for t in g.get("hints", []) if t not in tips]
        if tips:
            lines += ["", "Suggestions:", ""] + [f"- {t}" for t in tips]
    lines += ["", "_Scores are percentages (chrF++/BLEU on their native 0–100 scale; CER lower is better). "
              "CIs: percentile bootstrap over evaluated samples — with small `--limit`, read differences "
              "smaller than the CI width as noise._"]
    return "\n".join(lines) + "\n"


def write_reports(payload: Dict[str, Any], run_dir: str) -> None:
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(payload))
    rows = task_rows(payload)
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "family", "lang", "status", "n", "metric", "value", "ci_low", "ci_high",
                    "other_metrics", "seconds", "peak_gpu_mem_gb"])
        for r in rows:
            w.writerow([r["task"], r.get("family"), r["lang"], r["status"], r.get("n"), r.get("metric"),
                        r.get("value"), r.get("ci_low"), r.get("ci_high"),
                        json.dumps(r.get("others") or {}), r.get("seconds"), r.get("peak_mem")])


def print_summary(payload: Dict[str, Any]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover
        print(render_markdown(payload))
        return
    console = Console()
    t = Table(title=f"indibench — {payload.get('model', {}).get('name', '?')}", show_lines=False)
    for col, just in (("task", "left"), ("lang", "left"), ("n", "right"), ("metric", "left"),
                      ("score", "right"), ("95% CI", "left"), ("other", "left"), ("time", "right")):
        t.add_column(col, justify=just)
    for r in task_rows(payload):
        secs = fmt_duration(r.get("seconds")) if r.get("seconds") is not None else "–"
        if r["status"] in ("error", "skipped"):
            t.add_row(r["task"], r["lang"], "–", f"[red]{r['status']}[/red]", "–", "", r["note"][:70], secs)
            continue
        ci = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]" if r.get("ci_low") is not None else ""
        others = " ".join(f"{k}={_fmt(v)}" for k, v in (r.get("others") or {}).items())
        score = "predict-only" if r["status"] == "predict_only" else f"[bold]{_fmt(r.get('value'))}[/bold]"
        t.add_row(r["task"], r["lang"], str(r.get("n")), r.get("metric") or "", score, ci, others, secs)
    t.add_section()
    t.add_row("[bold]total[/bold]", "", "", "", "", "", "", f"[bold]{fmt_duration(_bench_seconds(payload))}[/bold]")
    console.print(t)
    console.print(_total_line(payload))
    langs, mat, metric_of = language_matrix(payload)
    if mat and len(langs) > 1:
        m = Table(title="Language matrix (primary metric)")
        m.add_column("family")
        for l in langs:
            m.add_column(l, justify="right")
        for fam, vals in mat.items():
            m.add_row(fam, *[_fmt(vals.get(l)) for l in langs])
        console.print(m)


def compare_runs(paths: List[str]) -> str:
    """Markdown table comparing the primary metric of several runs (results.json or run dirs)."""
    payloads = []
    for p in paths:
        f = os.path.join(p, "results.json") if os.path.isdir(p) else p
        with open(f, encoding="utf-8") as fh:
            payloads.append(json.load(fh))
    names = [pl.get("model", {}).get("name", f"run{i}") for i, pl in enumerate(payloads)]
    tasks: List[str] = []
    for pl in payloads:
        for t in pl.get("results", {}):
            if t not in tasks:
                tasks.append(t)
    header = "| task | metric | " + " | ".join(names) + (" | Δ (last−first) |" if len(payloads) > 1 else " |")
    lines = [header, "|---|---|" + "---:|" * (len(payloads) + (1 if len(payloads) > 1 else 0))]
    for t in tasks:
        vals, metric = [], ""
        for pl in payloads:
            res = pl.get("results", {}).get(t, {})
            pm, m = _primary(res)
            metric = metric or (pm or "")
            vals.append(m["value"] if m else None)
        row = f"| {t} | {metric} | " + " | ".join(_fmt(v) for v in vals)
        if len(payloads) > 1:
            d = (vals[-1] - vals[0]) if vals[0] is not None and vals[-1] is not None else None
            row += f" | {'' if d is None else ('+' if d >= 0 else '')}{_fmt(d)}"
        lines.append(row + " |")
    return "\n".join(lines) + "\n"
