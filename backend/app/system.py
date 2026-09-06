"""System information: GPU availability (nvidia-smi), TensorFlow CUDA build, CPU count, last training device."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from importlib import metadata
from typing import Any

from fastapi import APIRouter

from .config import DATA_DIR

router = APIRouter(prefix="/api", tags=["system"])
LAST_DEVICE_FILE = DATA_DIR / "last_training_device.json"
_cache: dict[str, Any] = {"at": 0.0, "value": None}


def _nvidia_smi() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(first) < 4:
        return None
    try:
        return {
            "name": first[0],
            "memory_total_mb": int(float(first[1])),
            "memory_used_mb": int(float(first[2])),
            "driver": first[3],
            "utilization": int(float(first[4])) if len(first) > 4 and first[4].isdigit() else None,
        }
    except ValueError:
        return None


def _tensorflow_cuda() -> bool:
    for pkg in ("nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"):
        try:
            metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return False
    return True


def system_info() -> dict[str, Any]:
    now = time.time()
    if _cache["value"] is not None and now - _cache["at"] < 10:
        return _cache["value"]
    gpu = _nvidia_smi()
    last: dict[str, Any] = {}
    if LAST_DEVICE_FILE.exists():
        try:
            last = json.loads(LAST_DEVICE_FILE.read_text())
        except json.JSONDecodeError:
            last = {}
    value = {
        "gpu": gpu,
        "gpu_available": gpu is not None,
        "tensorflow_cuda": _tensorflow_cuda(),
        "cpu_count": os.cpu_count(),
        "last_training": last or None,
    }
    _cache.update(at=now, value=value)
    return value


@router.get("/system")
def get_system():
    return system_info()
