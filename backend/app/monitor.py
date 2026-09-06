"""Saved activations from the long-run false-accept test."""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException

from .config import Project, current_project

router = APIRouter(prefix="/api", tags=["monitor"])
MONITOR_MAX_FILES = 200


def monitor_dir(project: Project) -> Path:
    path = project.dir / "monitor"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_detection(project: Project, pcm: bytes, job_id: str) -> dict[str, Any]:
    from .recordings import describe

    folder = monitor_dir(project)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
    path = folder / name
    audio = np.frombuffer(pcm, dtype=np.int16)
    sf.write(str(path), audio, 16000, subtype="PCM_16")
    old = sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    for stale in old[:-MONITOR_MAX_FILES]:
        stale.unlink(missing_ok=True)
    item = describe("monitor", path, url_prefix="/api/monitor/audio")
    item["url"] = f"/api/monitor/audio/{name}"
    item["job_id"] = job_id
    return item


# ----- monitor: saved detections from long-running tests ---------------------------------
SAFE_NAME = re.compile(r"^[a-zA-Z0-9_\-]+\.wav$")


@router.get("/monitor")
def list_monitor():
    from .recordings import describe

    project = current_project()
    folder = monitor_dir(project)
    items = []
    for path in sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        item = describe("monitor", path, url_prefix="/api/monitor/audio")
        item["url"] = f"/api/monitor/audio/{path.name}"
        items.append(item)
    return {"items": items}


@router.get("/monitor/audio/{name}")
def monitor_audio(name: str):
    if not SAFE_NAME.match(name):
        raise HTTPException(400, {"code": "bad_id", "message": "Bad name"})
    path = monitor_dir(current_project()) / name
    if not path.exists():
        raise HTTPException(404, {"code": "not_found", "message": "Not found"})
    from fastapi.responses import FileResponse

    return FileResponse(path, media_type="audio/wav", filename=name)


@router.post("/monitor/{name}/negative")
def monitor_to_negative(name: str):
    """Moves a saved false activation into the project's negative samples (tagged 'noisy' when it came from a monitor)."""
    from .recordings import describe, set_tag

    if not SAFE_NAME.match(name):
        raise HTTPException(400, {"code": "bad_id", "message": "Bad name"})
    project = current_project()
    src = monitor_dir(project) / name
    if not src.exists():
        raise HTTPException(404, {"code": "not_found", "message": "Not found"})
    dst = project.negative_dir / f"{name[:-4]}__monitor.wav"
    src.rename(dst)
    return describe("negative", dst)


@router.post("/monitor/adopt-all")
def monitor_adopt_all():
    """Moves every saved activation into the negative samples (used by "retrain with monitor negatives")."""
    project = current_project()
    folder = monitor_dir(project)
    moved = 0
    for src in sorted(folder.glob("*.wav")):
        src.rename(project.negative_dir / f"{src.name[:-4]}__monitor.wav")
        moved += 1
    return {"moved": moved}


@router.delete("/monitor/{name}")
def delete_monitor(name: str):
    if not SAFE_NAME.match(name):
        raise HTTPException(400, {"code": "bad_id", "message": "Bad name"})
    path = monitor_dir(current_project()) / name
    if path.exists():
        path.unlink()
    return {"deleted": name}


@router.delete("/monitor")
def clear_monitor():
    folder = monitor_dir(current_project())
    n = 0
    for path in folder.glob("*.wav"):
        path.unlink()
        n += 1
    return {"deleted": n}
