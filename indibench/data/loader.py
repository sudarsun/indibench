"""Dataset loading with sub-sampling suited to single-GPU runs.

Supports:
  * Hugging Face Hub datasets (``path`` + optional ``name``/``names``/``data_dir``)
  * raw files on the Hub or locally via ``data_files`` (json/jsonl/parquet/csv)
  * local files/directories (``path: ./my_data.jsonl``)
  * parallel corpora assembled from two configs (``pair:``) for translation
  * row filters, named preprocessors, streaming, deterministic sampling
"""
from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

log = logging.getLogger(__name__)

_FILE_FORMATS = {".json": "json", ".jsonl": "json", ".parquet": "parquet", ".csv": "csv", ".tsv": "csv"}


class DatasetLoadError(RuntimeError):
    pass


def _infer_format(files: Any) -> Optional[str]:
    if isinstance(files, dict):
        files = next(iter(files.values()))
    if isinstance(files, (list, tuple)):
        files = files[0] if files else ""
    ext = os.path.splitext(str(files))[1].lower()
    return _FILE_FORMATS.get(ext)


def _hf_token(env_name: str) -> Optional[str]:
    return os.environ.get(env_name) or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _load_raw(dspec: Dict[str, Any], split: str, streaming: bool, cache_dir: Optional[str], token: Optional[str]):
    """Return a ``datasets`` Dataset/IterableDataset for one (config, split)."""
    try:
        import datasets
    except ImportError as e:  # pragma: no cover
        raise DatasetLoadError("The 'datasets' package is required: pip install datasets") from e

    path = dspec.get("path")
    data_files = dspec.get("data_files")
    kwargs: Dict[str, Any] = {"cache_dir": cache_dir, "streaming": streaming}
    if token:
        kwargs["token"] = token
    if dspec.get("revision"):
        kwargs["revision"] = dspec["revision"]

    # Local file → use the generic builder.
    if path and os.path.isfile(os.path.expanduser(path)):
        data_files = os.path.expanduser(path)
        path = None
    if data_files is not None:
        fmt = dspec.get("format") or _infer_format(data_files)
        if not fmt:
            raise DatasetLoadError(f"Cannot infer file format from data_files={data_files!r}; set dataset.format")
        if isinstance(data_files, (str, list)):
            data_files = {split: data_files}
        if fmt == "csv" and str(next(iter(data_files.values()))).endswith(".tsv"):
            kwargs["delimiter"] = "\t"
        if dspec.get("json_field") and fmt == "json":
            kwargs["field"] = dspec["json_field"]
        return datasets.load_dataset(fmt, data_files=data_files, split=split, **kwargs)

    if not path:
        raise DatasetLoadError("dataset.path or dataset.data_files must be set")
    if os.path.isdir(os.path.expanduser(path)):
        path = os.path.expanduser(path)
    if dspec.get("name") is not None:
        kwargs["name"] = dspec["name"]
    if dspec.get("data_dir") is not None:
        kwargs["data_dir"] = dspec["data_dir"]
    return datasets.load_dataset(path, split=split, **kwargs)


def _matches(row: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    for col, want in flt.items():
        val = row.get(col)
        if want == "__notnull__":
            if val is None:
                return False
            continue
        if isinstance(want, list):
            if val not in want:
                return False
        elif val != want:
            return False
    return True


def _materialize(ds, flt: Optional[Dict[str, Any]], max_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in ds:
        if flt and not _matches(row, flt):
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def resolve_limit(limit: Optional[Union[int, float]], n_total: int) -> int:
    if limit is None:
        return n_total
    if isinstance(limit, float):
        return max(1, min(n_total, int(round(limit * n_total))))
    return min(n_total, int(limit))


def sample_indices(
    n_total: int,
    limit: Optional[Union[int, float]],
    mode: str = "random",
    seed: int = 1234,
    strata: Optional[Sequence[Any]] = None,
) -> List[int]:
    """Deterministically choose which rows to evaluate.

    ``stratified`` takes an equal share from every stratum (round-robin over shuffled
    strata), which keeps small runs representative across e.g. MILU subjects.
    """
    k = resolve_limit(limit, n_total)
    idx = list(range(n_total))
    if k >= n_total and mode != "random":
        return idx
    rng = random.Random(seed)
    if mode == "first":
        return idx[:k]
    if mode == "stratified" and strata is not None:
        buckets: Dict[Any, List[int]] = defaultdict(list)
        for i, s in enumerate(strata):
            buckets[s].append(i)
        keys = sorted(buckets, key=lambda x: str(x))
        for key in keys:
            rng.shuffle(buckets[key])
        out: List[int] = []
        pos = 0
        while len(out) < k:
            progressed = False
            for key in keys:
                b = buckets[key]
                if pos < len(b):
                    out.append(b[pos])
                    progressed = True
                    if len(out) >= k:
                        break
            if not progressed:
                break
            pos += 1
        return sorted(out)
    rng.shuffle(idx)
    return sorted(idx[:k])


class RowSource:
    """Lazily indexable rows: wraps a ``datasets.Dataset`` (images decoded only on access)
    or a plain list of dicts."""

    def __init__(self, data, root: Optional[str] = None):
        self._data = data
        self.root = root

    def __len__(self) -> int:
        return len(self._data)

    @property
    def is_hf(self) -> bool:
        return not isinstance(self._data, list)

    @property
    def column_names(self) -> List[str]:
        if self.is_hf:
            return list(self._data.column_names)
        return list(self._data[0].keys()) if self._data else []

    def column(self, name: str) -> List[Any]:
        if self.is_hf:
            return list(self._data[name])
        return [r.get(name) for r in self._data]

    def rows(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        if self.is_hf:
            sub = self._data.select(list(indices))
            return [dict(r) for r in sub]
        return [dict(self._data[i]) for i in indices]

    def features(self) -> Dict[str, str]:
        if self.is_hf:
            return {k: str(v) for k, v in self._data.features.items()}
        if not self._data:
            return {}
        return {k: type(v).__name__ for k, v in self._data[0].items()}


def load_rows(
    dspec: Dict[str, Any],
    split: Optional[str] = None,
    *,
    streaming: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    token_env: str = "HF_TOKEN",
    max_rows: Optional[int] = None,
) -> RowSource:
    """Load a dataset split as a :class:`RowSource`.

    Non-streaming Hub/local datasets stay Arrow-backed so that filtering, stratification
    and sub-sampling never decode images of rows that are not evaluated. Streaming mode
    (or any preprocessor) materialises rows as python dicts, capped at ``max_rows``.
    """
    from indibench.data.preprocess import apply_preprocessors

    split = split or dspec.get("split") or "test"
    streaming = bool(dspec.get("streaming", False)) if streaming is None else streaming
    token = _hf_token(token_env)
    flt = dspec.get("filter")

    root = None
    p = dspec.get("path")
    if p and os.path.exists(os.path.expanduser(p)):
        ep = os.path.expanduser(p)
        root = os.path.dirname(os.path.abspath(ep)) if os.path.isfile(ep) else os.path.abspath(ep)
    elif isinstance(dspec.get("data_files"), str) and os.path.exists(os.path.expanduser(dspec["data_files"])):
        root = os.path.dirname(os.path.abspath(os.path.expanduser(dspec["data_files"])))

    if dspec.get("pair"):
        return RowSource(_load_pair(dspec, split, streaming, cache_dir, token_env, max_rows), root)

    configs = dspec.get("names") or [dspec.get("name")]
    multi = bool(dspec.get("names"))
    parts = []
    try:
        for cfg_name in configs:
            sub = dict(dspec, name=cfg_name)
            sub.pop("names", None)
            ds = _load_raw(sub, split, streaming, cache_dir, token)
            if streaming:
                if max_rows is not None and dspec.get("shuffle_buffer"):
                    ds = ds.shuffle(seed=dspec.get("seed", 1234), buffer_size=int(dspec["shuffle_buffer"]))
                part = _materialize(ds, flt, max_rows)
                if multi:
                    for r in part:
                        r["_config"] = cfg_name
            else:
                if flt:
                    cols = list(flt)
                    ds = ds.filter(lambda *vals: _matches(dict(zip(cols, vals)), flt), input_columns=cols)
                if multi:
                    ds = ds.add_column("_config", [cfg_name] * len(ds))
                part = ds
            parts.append(part)
    except DatasetLoadError:
        raise
    except Exception as e:
        hint = ""
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            hint = (" -- access denied: if the dataset is gated, accept its terms on huggingface.co and "
                    f"export {token_env}=<your token>; otherwise check that huggingface.co is reachable")
        elif "script" in msg.lower() or "trust_remote_code" in msg.lower():
            hint = (" -- this dataset relies on a legacy loading script; point the task at raw files "
                    "with dataset.data_files (see README: 'Fixing a dataset').")
        raise DatasetLoadError(f"Failed to load {dspec.get('path') or dspec.get('data_files')} "
                               f"(name={dspec.get('name')}, split={split}): {msg}{hint}") from e

    if streaming:
        data: Any = [r for part in parts for r in part]
    elif len(parts) == 1:
        data = parts[0]
    else:
        import datasets
        try:
            data = datasets.concatenate_datasets(parts)
        except Exception:  # schema mismatch between configs -> fall back to python rows
            data = [dict(r) for part in parts for r in part]

    if dspec.get("preprocess"):
        rows = data if isinstance(data, list) else [dict(r) for r in data]
        data = apply_preprocessors(rows, dspec.get("preprocess"))
    return RowSource(data, root)


def _load_pair(dspec, split, streaming, cache_dir, token_env, max_rows) -> List[Dict[str, Any]]:
    """Zip two configs/files of an n-way parallel corpus into ``source``/``target`` columns."""
    pair = dspec["pair"]
    column = pair.get("column", "text")
    base = {k: v for k, v in dspec.items() if k not in ("pair", "preprocess", "filter")}
    src_spec = dict(base, **(pair.get("source") or {}))
    tgt_spec = dict(base, **(pair.get("target") or {}))
    src_src = load_rows(src_spec, split, streaming=streaming, cache_dir=cache_dir,
                        token_env=token_env, max_rows=max_rows)
    tgt_src = load_rows(tgt_spec, split, streaming=streaming, cache_dir=cache_dir,
                        token_env=token_env, max_rows=max_rows)
    src_rows = src_src.rows(range(len(src_src)))
    tgt_rows = tgt_src.rows(range(len(tgt_src)))
    key = pair.get("join_on")
    out = []
    if key:
        tgt_by = {r[key]: r for r in tgt_rows}
        for r in src_rows:
            t = tgt_by.get(r[key])
            if t is not None:
                out.append({"id": r[key], "source": r[column], "target": t[column]})
    else:
        if len(src_rows) != len(tgt_rows):
            log.warning("Parallel corpus sides differ in length (%d vs %d); truncating", len(src_rows), len(tgt_rows))
        for i, (s, t) in enumerate(zip(src_rows, tgt_rows)):
            out.append({"id": s.get("id", i), "source": s[column], "target": t[column]})
    return out
