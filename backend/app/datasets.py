"""Negative datasets: registry, background download with progress, status."""
from __future__ import annotations

import random
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import APIRouter, HTTPException

from .config import DATASETS_DIR

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"}

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

HF_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"

DATASETS: dict[str, dict[str, Any]] = {
    "mini_speech_commands": {
        "title": "Mini Speech Commands (Google)",
        "description": {
            "cs": "8 000 jednosekundových nahrávek běžné řeči (yes/no/up/down/…). Základní negativní dataset – stáhne se automaticky před prvním trénováním.",
            "en": "8,000 one-second recordings of common speech (yes/no/up/down/…). The base negative dataset – downloaded automatically before the first training run.",
        },
        "size_mb": 182,
        "required": True,
        "url": "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        "type": "audio",
        "folder": "mini_speech_commands",
    },
    "mit_rirs": {
        "title": "MIT room impulse responses",
        "description": {
            "cs": "271 impulzních odezev reálných místností (MIT Reverb survey). Nahrávky se při augmentaci „umístí“ do místnosti s dozvukem – model pak funguje i z dálky. Malé, stahuje se automaticky.",
            "en": "271 impulse responses of real rooms (MIT Reverb survey). Augmentation places your recordings into reverberant rooms so the model works from a distance. Small, downloaded automatically.",
        },
        "size_mb": 12,
        "required": True,
        "url": "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip",
        "type": "audio",
        "folder": "mit_rirs",
        "convert": {"max_seconds": 3.0, "max_files": None},
    },
    "esc50": {
        "title": "ESC-50 environmental sounds",
        "description": {
            "cs": "2 000 pětisekundových nahrávek zvuků domácnosti a prostředí (déšť, pes, vysavač, klávesnice…). Používá se jako reálné pozadí při augmentaci i jako negativa. Stažení ~600 MB, po převodu ~320 MB.",
            "en": "2,000 five-second clips of household and environmental sounds (rain, dog, vacuum cleaner, keyboard…). Used as real background during augmentation and as negatives. ~600 MB download, ~320 MB after conversion.",
        },
        "size_mb": 600,
        "required": False,
        "url": "https://github.com/karolpiczak/ESC-50/archive/master.zip",
        "type": "audio",
        "folder": "esc50",
        "convert": {"max_seconds": 5.0, "max_files": None},
    },
    "fma_xs": {
        "title": "FMA music (extra small)",
        "description": {
            "cs": "Krátké hudební ukázky (Free Music Archive) jako pozadí při augmentaci – rádio nebo televize v místnosti. Použije se 1 000 náhodných skladeb po 8 s.",
            "en": "Short music excerpts (Free Music Archive) used as background during augmentation – radio or TV in the room. 1,000 random tracks, 8 s each.",
        },
        "size_mb": 182,
        "required": False,
        "url": HF_ROOT.replace("kahrendt/microwakeword", "mchl914/fma_xsmall") + "fma_xs.zip",
        "type": "audio",
        "folder": "fma_16k",
        "convert": {"max_seconds": 8.0, "max_files": 1000},
    },
    "dinner_party_eval": {
        "title": "microWakeWord – dinner_party_eval",
        "description": {
            "cs": "Předpočítané spektrogramy hovoru více lidí v pozadí (jen pro validaci/test – měření falešných aktivací za hodinu).",
            "en": "Pre-computed spectrograms of background conversation (validation/test only – measures false accepts per hour).",
        },
        "size_mb": 82,
        "required": False,
        "url": HF_ROOT + "dinner_party_eval.zip",
        "type": "mmap",
        "folder": "dinner_party_eval",
    },
    "dinner_party": {
        "title": "microWakeWord – dinner_party",
        "description": {
            "cs": "Předpočítané spektrogramy hovoru více lidí – volitelný rozšiřující negativní dataset pro trénování.",
            "en": "Pre-computed spectrograms of background conversation – optional extra negative dataset for training.",
        },
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
        return any(p for p in path.rglob("*.wav") if not p.name.startswith("._"))
    return any(path.rglob("*_mmap"))


def convert_audio_tree(folder: Path, max_seconds: float | None, max_files: int | None, progress: Callable[[dict], None] | None = None) -> int:
    """Converts every audio file below ``folder`` to 16 kHz mono 16-bit WAV (in place, originals removed)."""
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT and not p.name.startswith("._")]
    if max_files and len(files) > max_files:
        random.Random(7).shuffle(files)
        for stale in files[max_files:]:
            stale.unlink(missing_ok=True)
        files = files[:max_files]
    done = 0
    lock = threading.Lock()

    def convert(src: Path) -> None:
        nonlocal done
        dst = src.with_suffix(".16k.wav")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)]
        ok = subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists() and dst.stat().st_size > 1000
        src.unlink(missing_ok=True)
        if ok:
            dst.rename(src.with_suffix(".wav"))
        else:
            dst.unlink(missing_ok=True)
        with lock:
            done += 1
            if progress and (done % 25 == 0 or done == len(files)):
                progress({"state": "converting", "received": done, "total": len(files)})

    with ThreadPoolExecutor(max_workers=max(2, min(8, (os_cpu_count() or 4)))) as pool:
        list(pool.map(convert, files))
    # remove leftovers (metadata, csv, non-audio files) but keep the wavs
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() != ".wav":
            p.unlink(missing_ok=True)
    return sum(1 for _ in folder.rglob("*.wav"))


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


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
        _set(name, received=None, total=None)
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
        for junk in target.rglob("__MACOSX"):
            shutil.rmtree(junk, ignore_errors=True)
        if meta.get("convert"):
            _set(name, state="converting", received=0, total=None)

            def conv_progress(info: dict) -> None:
                _set(name, received=info.get("received"), total=info.get("total"))
                if progress:
                    progress(info)

            convert_audio_tree(target, meta["convert"].get("max_seconds"), meta["convert"].get("max_files"), conv_progress)
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
        if current and current.get("state") in ("downloading", "extracting", "converting"):
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
