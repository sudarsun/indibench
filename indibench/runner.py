"""Evaluation loop: sample → prompt → batch inference → score → aggregate, with resume."""
from __future__ import annotations

import datetime as _dt
import logging
import os
import platform
import queue
import re
import threading
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

from indibench import __version__
from indibench.config import BenchConfig, dump_config
from indibench.data.loader import DatasetLoadError, sample_indices
from indibench.gpumon import GPUMonitor, utilisation_hints
from indibench.languages import expand_language_filter
from indibench.metrics import scores as M
from indibench.models.base import BaseModel, load_model
from indibench.tasks.base import Doc, Task, build_task
from indibench.tasks.registry import TaskRegistry
from indibench.utils import append_jsonl, read_jsonl, strip_thinking, write_json

log = logging.getLogger(__name__)


def _req_cost(r) -> int:
    return sum(len(p.get("text", "")) for m in r.messages for p in m["content"]) + 4000 * len(r.images)


def _chunks(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"indibench": __version__, "python": sys.version.split()[0],
                            "platform": platform.platform(), "time": _dt.datetime.now().isoformat()}
    for mod in ("torch", "transformers", "vllm", "datasets", "accelerate", "bitsandbytes"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "memory_gb": round(p.total_memory / 1024**3, 1),
                           "compute_capability": f"{p.major}.{p.minor}",
                           "count": torch.cuda.device_count(), "cuda": torch.version.cuda,
                           "bf16_native": (p.major, p.minor) >= (8, 0)}
            try:
                import subprocess

                drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                if drv:
                    info["gpu"]["driver"] = drv[0]
            except Exception:
                pass
    except Exception:
        pass
    return info


def _peak_mem_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        pass
    return None


def _reset_peak_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class Evaluator:
    def __init__(self, cfg: BenchConfig, model: Optional[BaseModel] = None):
        self.cfg = cfg
        self.registry = TaskRegistry(cfg.run.task_files)
        self.model = model
        self.languages = expand_language_filter(cfg.run.languages)
        self.task_names = self.registry.resolve(cfg.run.tasks, self.languages,
                                                modalities=cfg.run.modalities, exclude=cfg.run.exclude)
        if not self.task_names:
            raise ValueError("No tasks selected (check run.tasks / --languages / --modality / --exclude)")
        name = cfg.run.run_name or self._default_run_name()
        self.run_dir = os.path.join(cfg.run.output_dir, name)

    def _default_run_name(self) -> str:
        m = self.cfg.model.name_or_path or self.cfg.model.backend
        short = re.sub(r"[^A-Za-z0-9._-]+", "_", m.rstrip("/").split("/")[-1])
        return f"{short}__{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # ------------------------------------------------------------------ main
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        os.makedirs(os.path.join(self.run_dir, "predictions"), exist_ok=True)
        dump_config(cfg, os.path.join(self.run_dir, "config.yaml"))
        env = environment_info()
        write_json(os.path.join(self.run_dir, "env.json"), env)
        log.info("Run dir: %s", self.run_dir)
        log.info("Tasks (%d): %s", len(self.task_names), ", ".join(self.task_names))

        if self.model is None:
            from indibench.hardware import harden_config

            self.hardware_notes = harden_config(cfg)
            if self.hardware_notes:
                dump_config(cfg, os.path.join(self.run_dir, "config.yaml"))  # record adjusted dtype etc.
            t0 = time.time()
            self.model = load_model(cfg.model)
            log.info("Model ready in %.1fs", time.time() - t0)
        if not self.model.supports_images:
            vision = [n for n in self.task_names if self.registry.specs[n].get("modality") == "image"]
            if vision:
                log.warning("Model is text-only: skipping %d vision task(s): %s", len(vision), ", ".join(vision))
                self.task_names = [n for n in self.task_names if n not in vision]

        results: Dict[str, Any] = {}
        t_start = time.time()
        for name in self.task_names:
            spec = self.registry.get(name)
            try:
                task = build_task(spec, cfg.run, cfg.model.image)
                results[name] = self.run_task(task)
            except DatasetLoadError as e:
                log.error("[%s] dataset error: %s", name, e)
                results[name] = {"error": str(e), "family": spec.get("family"), "lang": spec.get("lang")}
            except KeyboardInterrupt:
                log.warning("Interrupted — partial predictions are saved; re-run with the same --run-name to resume")
                break
            except Exception as e:  # keep going; one broken task must not kill a long run
                log.exception("[%s] failed: %s", name, e)
                results[name] = {"error": f"{type(e).__name__}: {e}", "family": spec.get("family"),
                                 "lang": spec.get("lang")}
            self._write(results, env, time.time() - t_start)
        self._write(results, env, time.time() - t_start)
        payload = self._payload(results, env, time.time() - t_start)
        if cfg.run.executive_summary:
            try:
                from indibench.summary import write_executive_summary

                path = write_executive_summary(payload, self.run_dir)
                log.info("Executive summary: %s", path)
            except Exception as e:  # never fail a finished run because of the report
                log.warning("Could not build executive summary: %s", e)
        return payload

    def _payload(self, results, env, elapsed) -> Dict[str, Any]:
        return {"model": self.model.info() if self.model else {"name": self.cfg.model.name_or_path},
                "config": self.cfg.to_dict(), "env": env, "elapsed_s": round(elapsed, 1),
                "hardware_notes": getattr(self, "hardware_notes", []), "results": results}

    def _write(self, results, env, elapsed) -> None:
        from indibench.report import write_reports

        payload = self._payload(results, env, elapsed)
        write_json(os.path.join(self.run_dir, "results.json"), payload)
        write_reports(payload, self.run_dir)

    # ------------------------------------------------------------------ one task
    def run_task(self, task: Task) -> Dict[str, Any]:
        cfg, model = self.cfg, self.model
        assert model is not None
        if task.modality == "image" and not model.supports_images:
            log.warning("[%s] skipped: model has no image support", task.name)
            return {"skipped": "model has no image support", "family": task.family, "lang": task.lang}

        limit = cfg.run.limit
        streaming_cap = None
        if (cfg.run.streaming or task.dataset_spec().get("streaming")) and isinstance(limit, int):
            streaming_cap = limit * 4 if cfg.run.sampling != "first" else limit
        t0 = time.time()
        src = task.load_source(max_rows=streaming_cap)
        n_total = len(src)
        if n_total == 0:
            raise DatasetLoadError(f"{task.name}: dataset is empty after filtering")
        strata = None
        if cfg.run.sampling == "stratified":
            col = task.strata_column()
            if col and col in src.column_names:
                strata = src.column(col)
            else:
                log.warning("[%s] stratified sampling requested but column %r unavailable; using random",
                            task.name, col)
        mode = cfg.run.sampling
        if mode == "stratified" and strata is None:
            mode = "random"
        idx = sample_indices(n_total, limit, mode, cfg.run.seed, strata)
        log.info("[%s] %d/%d samples selected (load %.1fs)", task.name, len(idx), n_total, time.time() - t0)

        pred_path = os.path.join(self.run_dir, "predictions", f"{task.name}.jsonl")
        if not cfg.run.resume and os.path.exists(pred_path):
            os.remove(pred_path)
        done = {r["id"]: r for r in read_jsonl(pred_path)} if cfg.run.resume else {}

        use_ll = task.wants_loglikelihood()
        if use_ll and not model.supports_loglikelihood:
            log.warning("[%s] backend has no log-likelihood support; using generation", task.name)
            use_ll = False

        records: List[Dict[str, Any]] = []
        skipped = 0
        logged = 0
        bs = self._batch_size(task)
        chunk = model.inference_chunk(bs, task.modality)
        if cfg.run.inference_chunk:
            chunk = int(cfg.run.inference_chunk)
        prep_group = max(chunk, bs * 8)           # a pool large enough to length-sort into batches
        model.reset_capacity()
        _reset_peak_mem()
        t_inf = time.time()
        n_new = 0
        progress = _Progress(task.name, len(idx))
        state = {"use_ll": use_ll}

        # Producer: decode rows, build docs and model requests (image decode/resize, prompt
        # rendering) in a background thread so the GPU never waits for the CPU between batches.
        q: "queue.Queue" = queue.Queue(maxsize=max(1, int(cfg.run.prefetch)))
        stop = threading.Event()

        def produce():
            try:
                for part in _chunks(idx, prep_group):
                    if stop.is_set():
                        return
                    docs, reqs, resumed, bad = [], [], [], 0
                    for i, row in zip(part, src.rows(part)):
                        try:
                            d = task.make_doc(row, i)
                        except Exception as e:
                            log.debug("[%s] bad row %s: %s", task.name, i, e)
                            d = None
                        if d is None:
                            bad += 1
                        elif d.id in done:
                            resumed.append(done[d.id])
                        else:
                            docs.append(d)
                            reqs.append(self._build(task, d, state["use_ll"]))
                    q.put(("group", docs, reqs, resumed, bad))
                q.put(("end", None, None, None, 0))
            except BaseException as e:  # surface producer errors in the main thread
                q.put(("error", e, None, None, 0))

        producer = threading.Thread(target=produce, name=f"prefetch-{task.name}", daemon=True)
        monitor = GPUMonitor().start() if cfg.run.gpu_monitor else None
        producer.start()
        try:
            while True:
                kind, docs, reqs, resumed, bad = q.get()
                if kind == "error":
                    raise docs
                if kind == "end":
                    break
                skipped += bad
                records.extend(resumed)
                progress.advance(len(resumed))
                # Similar-length prompts together -> less padding for static-batching backends.
                order = sorted(range(len(docs)), key=lambda k: _req_cost(reqs[k]))
                docs, reqs = [docs[k] for k in order], [reqs[k] for k in order]
                for a in range(0, len(docs), chunk):
                    bd, br = docs[a:a + chunk], reqs[a:a + chunk]
                    try:
                        new = self._run_batch(task, bd, br, state["use_ll"])
                    except NotImplementedError as e:
                        log.warning("[%s] %s — falling back to generation", task.name, e)
                        state["use_ll"] = False
                        br = [self._build(task, d, False) for d in bd]
                        new = self._run_batch(task, bd, br, False)
                    for d, rec in zip(bd, new):
                        if logged < cfg.run.log_samples:
                            log.info("[%s] sample %s\n--- prompt ---\n%s\n--- response ---\n%s\n--- pred=%r gold=%r",
                                     task.name, d.id, rec["prompt"][-1500:], rec["response"][:500], rec["pred"],
                                     rec["gold"])
                            logged += 1
                    records.extend(new)
                    n_new += len(new)
                    if cfg.run.save_predictions:
                        append_jsonl(pred_path, new)
                    progress.advance(len(new))
        finally:
            stop.set()
            while producer.is_alive():          # unblock a producer waiting on a full queue
                try:
                    q.get_nowait()
                except queue.Empty:
                    producer.join(timeout=0.1)
            gpu_stats = monitor.stop() if monitor else None
        use_ll = state["use_ll"]
        progress.close()
        inf_time = time.time() - t_inf

        if not records:
            mapped = {k: v for k, v in task.fields.items() if isinstance(v, (str, list))}
            raise DatasetLoadError(f"{task.name}: no usable samples (all {skipped} rows were rejected). "
                                   f"Dataset columns: {src.column_names}; task field mapping: {mapped}. "
                                   f"Fix `fields` (see `indibench inspect {task.name}`).")
        metrics = {} if task.predict_only else task.aggregate(records, cfg.run.bootstrap)
        primary = task.primary_metric()
        out: Dict[str, Any] = {
            "family": task.family, "lang": task.lang, "type": task.type_name, "modality": task.modality,
            "tags": task.tags, "n": len(records), "n_total": n_total, "n_skipped": skipped,
            "primary_metric": primary, "metrics": metrics, "predict_only": task.predict_only,
            "mode": "loglikelihood" if use_ll else "generate",
            "timing": {"seconds": round(inf_time, 1),
                       "samples_per_s": round(n_new / inf_time, 3) if inf_time > 0 and n_new else None},
            "peak_gpu_mem_gb": _peak_mem_gb(),
            "batch_size": bs, "inference_chunk": chunk,
        }
        if gpu_stats:
            out["gpu"] = gpu_stats
            hints = utilisation_hints(gpu_stats, cfg.model.backend, bs, task.modality)
            if hints:
                out["gpu"]["hints"] = hints
                log.warning("[%s] GPU busy only %.0f%% of the time (memory peak %.0f%%). Try: %s", task.name,
                            gpu_stats["util_avg"], gpu_stats.get("mem_peak_pct") or 0, "; ".join(hints))
            else:
                log.info("[%s] GPU utilisation %.0f%%, memory peak %.0f%%", task.name, gpu_stats["util_avg"],
                         gpu_stats.get("mem_peak_pct") or 0)
        if task.group_by and not task.predict_only:
            out["breakdown"] = self._breakdown(task, records, primary)
        nch = [r["n_choices"] for r in records if r.get("n_choices")]
        if nch and primary in ("accuracy",):
            out["chance"] = round(100.0 * sum(1.0 / n for n in nch) / len(nch), 2)   # random-guess baseline
        out["description"] = (task.spec.get("description") or "").strip()
        if metrics.get(primary):
            pm = metrics[primary]
            ci = f" [{pm['ci_low']:.1f}, {pm['ci_high']:.1f}]" if pm.get("ci_low") is not None else ""
            log.info("[%s] %s = %.2f%s (n=%d)", task.name, primary, pm["value"], ci, len(records))
        return out

    def _batch_size(self, task: Task) -> int:
        bs = self.cfg.run.batch_size
        if bs == "auto":
            # Start big; the hf backend halves on OOM and remembers the size that fits.
            return 16 if task.modality == "image" else 64
        return int(bs)

    def _build(self, task: Task, d: Doc, use_ll: bool):
        sys_p = self.cfg.model.system_prompt
        if use_ll:
            return task.build_ll_request(d, sys_p)
        r = task.build_request(d, sys_p)
        try:
            r._gold = task.gold_text(d)  # used only by the dummy 'oracle' backend
        except Exception:
            r._gold = None
        return r

    def _run_batch(self, task: Task, docs: List[Doc], reqs: list, use_ll: bool) -> List[Dict[str, Any]]:
        cfg, model = self.cfg, self.model
        if use_ll:
            outs = model.loglikelihood(reqs)
            scored = [task.score_ll(d, o) for d, o in zip(docs, outs)]
            responses = [str([round(x[0], 3) for x in o]) for o in outs]
        else:
            raw = model.generate(reqs, cfg.generation)
            responses = [strip_thinking(x) if cfg.generation.strip_thinking else x for x in raw]
            if getattr(task, "parallel_scoring", False) and len(docs) > 1:
                # e.g. code execution: each sample runs in its own subprocess
                from concurrent.futures import ThreadPoolExecutor

                workers = max(1, min(len(docs), int(cfg.run.score_workers or (os.cpu_count() or 2))))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    scored = list(ex.map(lambda dx: task.score(dx[0], dx[1]), zip(docs, responses)))
            else:
                scored = [task.score(d, x) for d, x in zip(docs, responses)]
        out = []
        for d, r, resp, (pred, mets) in zip(docs, reqs, responses, scored):
            prompt_text = "\n".join(p["text"] for p in r.messages[-1]["content"] if p.get("type") == "text")
            rec = {"id": d.id, "prompt": prompt_text, "n_images": len(r.images), "response": resp,
                   "pred": pred, "gold": task.gold_repr(d), "metrics": mets, "group": d.group}
            if d.choices:
                rec["n_choices"] = len(d.choices)
            out.append(rec)
        return out

    def _breakdown(self, task: Task, records: List[Dict[str, Any]], primary: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in task.group_by:
            buckets: Dict[str, List[float]] = {}
            for r in records:
                g = (r.get("group") or {}).get(key)
                v = (r.get("metrics") or {}).get(primary)
                if g is not None and v is not None:
                    buckets.setdefault(g, []).append(v)
            if buckets:
                out[key] = {g: {"value": 100 * M.mean_ci(v, 0)[0], "n": len(v)} for g, v in sorted(buckets.items())}
        return out


class _Progress:
    """Thin rich progress bar that degrades to periodic log lines."""

    def __init__(self, name: str, total: int):
        self.total = total
        self.count = 0
        self._p = None
        try:
            from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

            if sys.stderr.isatty():
                self._p = Progress(TextColumn(f"[bold]{name}"), BarColumn(), MofNCompleteColumn(),
                                   TimeRemainingColumn(), transient=True)
                self._p.start()
                self._t = self._p.add_task(name, total=total)
        except Exception:
            self._p = None
        self._name = name
        self._last = 0.0

    def advance(self, n: int = 1) -> None:
        self.count += n
        if self._p is not None:
            self._p.advance(self._t, n)
        elif time.time() - self._last > 30:
            log.info("[%s] %d/%d", self._name, self.count, self.total)
            self._last = time.time()

    def close(self) -> None:
        if self._p is not None:
            self._p.stop()
