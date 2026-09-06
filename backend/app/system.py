"""System information: GPU availability (nvidia-smi), TensorFlow CUDA build, CPU count, last training device."""
from __future__ import annotations

import json
import os
from pathlib import Path
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
_update_cache: dict[str, Any] = {"at": 0.0, "value": None}
REPO = os.environ.get("GITHUB_REPO", "FilipChalupa/wakework-trainer")


def app_version() -> str:
    env = os.environ.get("APP_VERSION", "").strip()
    if env:
        return env
    for candidate in (Path(__file__).resolve().parent.parent / "VERSION", Path(__file__).resolve().parent.parent.parent / "VERSION"):
        if candidate.is_file():
            return candidate.read_text().strip()
    return "dev"


def _version_tuple(text: str) -> tuple[int, ...]:
    out = []
    for piece in text.lstrip("v").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out[:3])


def latest_release() -> dict[str, Any] | None:
    """Newest tag on GitHub (cached for 6 hours; failures are silent – offline installs are fine)."""
    now = time.time()
    if now - _update_cache["at"] < 6 * 3600:
        return _update_cache["value"]
    _update_cache["at"] = now
    try:
        import requests

        res = requests.get(f"https://api.github.com/repos/{REPO}/tags?per_page=5", timeout=5, headers={"Accept": "application/vnd.github+json"})
        tags = [t["name"] for t in res.json()] if res.ok else []
        versions = sorted((t for t in tags if _version_tuple(t) > (0,)), key=_version_tuple, reverse=True)
        _update_cache["value"] = {"latest": versions[0], "url": f"https://github.com/{REPO}/releases"} if versions else None
    except Exception:  # noqa: BLE001
        _update_cache["value"] = None
    return _update_cache["value"]


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
    current = app_version()
    release = latest_release()
    value = {
        "version": current,
        "latest_version": release["latest"] if release else None,
        "update_available": bool(release and _version_tuple(release["latest"]) > _version_tuple(current)),
        "releases_url": release["url"] if release else f"https://github.com/{REPO}",
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
