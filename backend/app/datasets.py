"""Negative datasets: registry, background download with progress, status."""
from __future__ import annotations

import shutil
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import APIRouter, HTTPException

from .config import DATASETS_DIR

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

HF_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"

DATASETS: dict[str, dict[str, Any]] = {
    "mini_speech_commands": {
        "title": "Mini Speech Commands (Google)",
        "description": "8 000 jednosekundových nahrávek běžné řeči (yes/no/up/down/…). Základní negativní dataset – stáhne se automaticky před prvním trénováním.",
        "size_mb": 182,
        "required": True,
        "url": "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        "type": "audio",
        "folder": "mini_speech_commands",
    },
    "dinner_party_eval": {
        "title": "microWakeWord – dinner_party_eval",
        "description": "Předpočítané spektrogramy hovoru více lidí v pozadí (jen pro validaci/test – měření falešných aktivací za hodinu).",
        "size_mb": 82,
        "required": False,
        "url": HF_ROOT + "dinner_party_eval.zip",
        "type": "mmap",
        "folder": "dinner_party_eval",
    },
    "dinner_party": {
        "title": "microWakeWord – dinner_party",
        "description": "Předpočítané spektrogramy hovoru více lidí – volitelný rozšiřující negativní dataset pro trénování.",
        "size_mb": 444,
        "required": False,
        "url": HF_ROOT + "dinner_party.zip",
        "type": "mmap",
        "folder": "dinner_party",
    },
}

_downloads: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def dataset_path(name: str) -> Path:
    return DATASETS_DIR / DATASETS[name]["folder"]


def is_installed(name: str) -> bool:
    path = dataset_path(name)
    if not path.is_dir():
        return False
    if DATASETS[name]["type"] == "audio":
        return any(path.rglob("*.wav"))
    return any(path.rglob("*_mmap"))


def status() -> list[dict[str, Any]]:
    out = []
    for name, meta in DATASETS.items():
        entry = {
            "id": name,
            "title": meta["title"],
            "description": meta["description"],
            "size_mb": meta["size_mb"],
            "required": meta["required"],
            "installed": is_installed(name),
            "download": _downloads.get(name),
        }
        out.append(entry)
    return out


def _set(name: str, **fields):
    with _lock:
        _downloads.setdefault(name, {}).update(fields)


def download(name: str, progress: Callable[[dict], None] | None = None) -> None:
    """Blocking download + extract. Safe to call from a thread."""
    meta = DATASETS[name]
    target = dataset_path(name)
    tmp_zip = DATASETS_DIR / f"{meta['folder']}.zip.part"
    _set(name, state="downloading", received=0, total=None, error=None)
    try:
        with requests.get(meta["url"], stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0) or None
            received = 0
            with open(tmp_zip, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    received += len(chunk)
                    _set(name, received=received, total=total)
                    if progress:
                        progress({"received": received, "total": total})
        _set(name, state="extracting")
        if progress:
            progress({"state": "extracting"})
        if target.exists():
            shutil.rmtree(target)
        with zipfile.ZipFile(tmp_zip) as zf:
            names = zf.namelist()
            top_levels = {n.split("/")[0] for n in names if n.strip("/")}
            if len(top_levels) == 1 and top_levels == {meta["folder"]}:
                zf.extractall(DATASETS_DIR)
            else:
                zf.extractall(target)
        tmp_zip.unlink(missing_ok=True)
        if not is_installed(name):
            raise RuntimeError("Archive extracted but expected files were not found")
        _set(name, state="done")
    except Exception as exc:  # noqa: BLE001
        tmp_zip.unlink(missing_ok=True)
        _set(name, state="error", error=str(exc))
        raise


def start_download(name: str) -> dict[str, Any]:
    if name not in DATASETS:
        raise HTTPException(404, "Unknown dataset")
    with _lock:
        current = _downloads.get(name)
        if current and current.get("state") in ("downloading", "extracting"):
            return current
    threading.Thread(target=_safe_download, args=(name,), daemon=True).start()
    return _downloads.get(name, {"state": "downloading"})


def _safe_download(name: str):
    try:
        download(name)
    except Exception:  # noqa: BLE001
        pass


@router.get("")
def get_datasets():
    return {"items": status(), "dir": str(DATASETS_DIR)}


@router.post("/{name}/download")
def post_download(name: str):
    return start_download(name)


@router.delete("/{name}")
def delete_dataset(name: str):
    if name not in DATASETS:
        raise HTTPException(404, "Unknown dataset")
    path = dataset_path(name)
    if path.exists():
        shutil.rmtree(path)
    with _lock:
        _downloads.pop(name, None)
    return {"deleted": name}
