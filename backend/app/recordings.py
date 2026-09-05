"""Upload / list / play / delete recorded samples (+ quality analysis, trash with restore)."""
from __future__ import annotations

import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .audio import normalize_wav
from .config import NEGATIVE_DIR, POSITIVE_DIR

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

KINDS = {"positive": POSITIVE_DIR, "negative": NEGATIVE_DIR}
SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]+\.wav$")
TRASH_KEEP = 50
PEAK_BUCKETS = 48

_analysis_cache: dict[tuple[str, int], dict] = {}
_cache_lock = threading.Lock()


def _dir(kind: str) -> Path:
    if kind not in KINDS:
        raise HTTPException(400, f"Unknown kind '{kind}'")
    return KINDS[kind]


def _trash(kind: str) -> Path:
    path = _dir(kind) / ".trash"
    path.mkdir(exist_ok=True)
    return path


def analyze(path: Path) -> dict:
    """Peak-envelope for a mini waveform + simple quality heuristics."""
    key = (str(path), int(path.stat().st_mtime_ns))
    with _cache_lock:
        cached = _analysis_cache.get(key)
    if cached:
        return cached
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:  # noqa: BLE001
        return {"duration": 0.0, "peaks": [], "quality": {"issues": ["unreadable"]}}
    n = audio.shape[0]
    duration = n / sr if sr else 0.0
    peaks: list[float] = []
    if n:
        size = max(1, n // PEAK_BUCKETS)
        usable = audio[: size * PEAK_BUCKETS]
        if usable.size:
            peaks = np.abs(usable.reshape(-1, size)).max(axis=1).round(3).tolist()
    peak = float(np.max(np.abs(audio))) if n else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if n else 0.0
    issues: list[str] = []
    speech_start = speech_end = None
    if n:
        frame = max(1, sr // 100)
        frames = audio[: (n // frame) * frame].reshape(-1, frame)
        if frames.size:
            frame_rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
            threshold = max(frame_rms.max() * 10 ** (-30 / 20), 0.004)
            active = np.where(frame_rms > threshold)[0]
        else:
            active = np.array([], dtype=int)
        if active.size:
            speech_start = float(active[0] * frame / sr)
            speech_end = float((active[-1] + 1) * frame / sr)
            edge = 0.08
            if speech_start < edge:
                issues.append("cut_start")
            if duration - speech_end < edge:
                issues.append("cut_end")
            if speech_end - speech_start < 0.15:
                issues.append("too_short")
        else:
            issues.append("silent")
    if peak >= 0.985:
        issues.append("clipping")
    elif peak < 0.08 and "silent" not in issues:
        issues.append("too_quiet")
    result = {
        "duration": round(duration, 3),
        "peaks": peaks,
        "quality": {
            "peak": round(peak, 3),
            "rms_db": round(20 * np.log10(rms + 1e-9), 1),
            "speech_start": speech_start,
            "speech_end": speech_end,
            "issues": issues,
        },
    }
    with _cache_lock:
        if len(_analysis_cache) > 2000:
            _analysis_cache.clear()
        _analysis_cache[key] = result
    return result


def _describe(kind: str, path: Path) -> dict:
    stat = path.stat()
    info = analyze(path)
    return {
        "id": path.name,
        "kind": kind,
        "duration": info["duration"],
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "url": f"/api/recordings/{kind}/{path.name}",
        "peaks": info["peaks"],
        "quality": info["quality"],
    }


def list_recordings(kind: str) -> list[dict]:
    files = sorted(_dir(kind).glob("*.wav"), key=lambda p: (p.stat().st_mtime, p.name))
    return [_describe(kind, p) for p in files]


def _check_id(rec_id: str) -> None:
    if not SAFE_ID.match(rec_id):
        raise HTTPException(400, "Bad id")


@router.get("")
def get_recordings(kind: str = "positive"):
    items = list_recordings(kind)
    return {"kind": kind, "items": items, "count": len(items)}


@router.get("/counts")
def get_counts():
    return {k: len(list(d.glob("*.wav"))) for k, d in KINDS.items()}


@router.post("")
async def upload_recording(file: UploadFile = File(...), kind: str = Form("positive")):
    target_dir = _dir(kind)
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")
    try:
        wav, duration = normalize_wav(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    if duration < 0.3:
        raise HTTPException(400, "Recording is too short")
    if duration > 15:
        raise HTTPException(400, "Recording is too long (max 15 s)")
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
    path = target_dir / name
    path.write_bytes(wav)
    return _describe(kind, path)


@router.get("/{kind}/{rec_id}")
def get_recording(kind: str, rec_id: str):
    _check_id(rec_id)
    path = _dir(kind) / rec_id
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav", filename=rec_id)


@router.delete("/{kind}/{rec_id}")
def delete_recording(kind: str, rec_id: str):
    """Soft delete: the file is moved to a trash folder so it can be restored."""
    _check_id(rec_id)
    path = _dir(kind) / rec_id
    if not path.exists():
        raise HTTPException(404, "Not found")
    trash = _trash(kind)
    path.rename(trash / rec_id)
    old = sorted(trash.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    for stale in old[:-TRASH_KEEP]:
        stale.unlink(missing_ok=True)
    return {"deleted": rec_id, "restorable": True}


@router.post("/{kind}/{rec_id}/restore")
def restore_recording(kind: str, rec_id: str):
    _check_id(rec_id)
    src = _trash(kind) / rec_id
    if not src.exists():
        raise HTTPException(404, "Not in trash")
    dst = _dir(kind) / rec_id
    src.rename(dst)
    return _describe(kind, dst)
