"""Small shared helpers: images, logging, JSON IO, think-tag stripping."""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("indibench")


def setup_logging(level: str = "INFO") -> None:
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
    except ImportError:  # pragma: no cover
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt,
                        datefmt="[%X]", handlers=[handler], force=True)
    for noisy in ("urllib3", "httpx", "filelock", "fsspec", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------ images
def to_pil(obj: Any, root: Optional[str] = None):
    """Convert dataset image values (PIL, {'bytes','path'}, path str, raw bytes) to RGB PIL."""
    from PIL import Image

    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        img = obj
    elif isinstance(obj, dict):
        if obj.get("bytes"):
            img = Image.open(io.BytesIO(obj["bytes"]))
        elif obj.get("path"):
            return to_pil(obj["path"], root)
        else:
            return None
    elif isinstance(obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(obj))
    elif isinstance(obj, str):
        if obj.startswith(("http://", "https://")):
            raise ValueError("Remote image URLs are not fetched; download images locally first")
        path = obj if os.path.isabs(obj) or root is None else os.path.join(root, obj)
        img = Image.open(os.path.expanduser(path))
    else:
        raise TypeError(f"Unsupported image value of type {type(obj).__name__}")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_to_budget(img, max_pixels: Optional[int], min_pixels: Optional[int] = None):
    """Downscale (or upscale) preserving aspect ratio so that w*h is within the pixel budget.

    Bounding pixels bounds the number of visual tokens, which is the main memory lever
    for VLMs on a single 40–48 GB GPU.
    """
    if img is None:
        return None
    w, h = img.size
    area = w * h
    if max_pixels and area > max_pixels:
        s = math.sqrt(max_pixels / area)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    elif min_pixels and area < min_pixels:
        s = math.sqrt(min_pixels / area)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    return img


def pil_to_data_uri(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt in ("JPG", "JPEG"):
        img.save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ------------------------------------------------------------------ text
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    # Unclosed think block (generation truncated mid-thought) or a dangling close tag.
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    elif re.search(r"<think>", text, re.IGNORECASE):
        text = ""
    return text.strip()


class SafeDict(dict):
    """``str.format_map`` helper that leaves unknown placeholders intact."""

    def __missing__(self, key):
        return "{" + key + "}"


def fmt(template: Any, variables: Dict[str, Any]) -> Any:
    """Recursively format strings inside nested lists/dicts with ``variables``."""
    if isinstance(template, str):
        return template.format_map(SafeDict(variables))
    if isinstance(template, list):
        return [fmt(t, variables) for t in template]
    if isinstance(template, dict):
        return {k: fmt(v, variables) for k, v in template.items()}
    return template


# ------------------------------------------------------------------ io
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping corrupt line in %s", path)
    return out


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
