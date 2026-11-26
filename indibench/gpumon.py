"""Background GPU utilisation / memory sampler (NVML if available, else ``nvidia-smi``).

Used by the runner to record, per task, how busy the GPU actually was — the quickest way to
tell whether a run is GPU-bound (good) or starved by batching / CPU preprocessing (fixable).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional


def _visible_indices() -> Optional[List[int]]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not v:
        return None
    try:
        return [int(x) for x in v.split(",") if x.strip() != ""]
    except ValueError:  # UUID / MIG identifiers: just sample everything
        return None


class GPUMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: List[List[tuple]] = []      # per tick: [(util %, mem used GiB, mem total GiB), ...]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handles: List[Any] = []
        try:
            import pynvml  # provided by the `nvidia-ml-py` package

            pynvml.nvmlInit()
            idx = _visible_indices() or list(range(pynvml.nvmlDeviceGetCount()))
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]
            self._nvml = pynvml
        except Exception:
            self._nvml = None
        self.available = bool(self._nvml) or shutil.which("nvidia-smi") is not None

    def _sample(self) -> List[tuple]:
        if self._nvml:
            out = []
            for h in self._handles:
                u = self._nvml.nvmlDeviceGetUtilizationRates(h)
                m = self._nvml.nvmlDeviceGetMemoryInfo(h)
                out.append((float(u.gpu), m.used / 1024**3, m.total / 1024**3))
            return out
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        vis = _visible_indices()
        out = []
        for line in r.stdout.strip().splitlines():
            try:
                i, u, used, tot = [x.strip() for x in line.split(",")]
                if vis is None or int(i) in vis:
                    out.append((float(u), float(used) / 1024, float(tot) / 1024))
            except ValueError:
                continue
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = self._sample()
                if s:
                    self.samples.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> "GPUMonitor":
        if self.available:
            self.samples = []
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gpumon", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> Optional[Dict[str, Any]]:
        if not self._thread:
            return None
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None
        return summarise(self.samples)


def summarise(samples: List[List[tuple]]) -> Optional[Dict[str, Any]]:
    if len(samples) < 2:
        return None
    n_gpu = len(samples[0])
    util = [sum(s[g][0] for g in range(n_gpu)) / n_gpu for s in samples if len(s) == n_gpu]
    mem = [max(s[g][1] for g in range(n_gpu)) for s in samples if len(s) == n_gpu]
    total = samples[0][0][2]
    return {
        "util_avg": round(sum(util) / len(util), 1),
        "util_p50": round(sorted(util)[len(util) // 2], 1),
        "idle_share": round(100 * sum(1 for u in util if u < 10) / len(util), 1),   # % of time near idle
        "mem_peak_gb": round(max(mem), 2),
        "mem_total_gb": round(total, 2),
        "mem_peak_pct": round(100 * max(mem) / total, 1) if total else None,
        "n_gpus": n_gpu,
        "seconds": len(util),
    }


def utilisation_hints(stats: Optional[Dict[str, Any]], backend: str, batch_size: Any, modality: str) -> List[str]:
    """Plain-language advice when a task left the GPU under-used."""
    if not stats or stats.get("seconds", 0) < 10:
        return []
    hints: List[str] = []
    util, mem = stats["util_avg"], stats.get("mem_peak_pct") or 0
    if util >= 70:
        return hints
    if backend == "hf":
        if mem < 60:
            hints.append(f"memory only {mem:.0f}% used: raise --batch-size (or use --batch-size auto)")
        hints.append("transformers batches wait for the longest answer; --backend vllm usually doubles to "
                     "10x throughput on generation-heavy tasks")
    elif backend == "vllm":
        hints.append("vLLM is starved: raise --inference-chunk 2048, and check CPU-side image "
                     "decoding (lower --max-pixels) or tokenisation is not the bottleneck")
    elif backend == "openai":
        hints.append("raise --concurrency (e.g. 64-128) so the server always has a full batch")
    if modality == "image" and stats.get("idle_share", 0) > 30:
        hints.append("frequent idle gaps on an image task suggest CPU-bound image loading; lower "
                     "--max-pixels or raise --prefetch 4")
    return hints
