"""Paths, multi-project layout and persisted project settings.

Layout of the data volume:
    /data/negative_datasets/        shared downloaded datasets
    /data/features_cache/           shared pre-computed spectrograms + synthetic noise
    /data/projects/<id>/project.json, positive_samples/, negative_samples/, jobs/
    /data/current_project           id of the project the UI works with
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")).resolve()
DATASETS_DIR = DATA_DIR / "negative_datasets"
FEATURE_CACHE_DIR = DATA_DIR / "features_cache"
PROJECTS_DIR = DATA_DIR / "projects"
CURRENT_FILE = DATA_DIR / "current_project"
KEEP_JOBS = int(os.environ.get("KEEP_JOBS", "10"))

DEFAULT_TRAINING = {
    "training_steps": 4000,
    "learning_rate": 0.001,
    "batch_size": 128,
    "eval_step_interval": 250,
    "augmentations_per_sample": 40,
    "clip_duration_ms": 1500,
    "negative_class_weight": 10.0,
    "positive_class_weight": 1.0,
    "hard_negatives": True,
}

_lock = threading.RLock()
PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text or "wakeword"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Project:
    id: str

    @property
    def dir(self) -> Path:
        return PROJECTS_DIR / self.id

    @property
    def file(self) -> Path:
        return self.dir / "project.json"

    @property
    def positive_dir(self) -> Path:
        return self.dir / "positive_samples"

    @property
    def negative_dir(self) -> Path:
        return self.dir / "negative_samples"

    @property
    def jobs_dir(self) -> Path:
        return self.dir / "jobs"

    def ensure(self) -> "Project":
        for d in (self.positive_dir, self.negative_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def sample_dir(self, kind: str) -> Path:
        if kind == "positive":
            return self.positive_dir
        if kind == "negative":
            return self.negative_dir
        raise KeyError(kind)


def _default_settings(name: str, wake_word: str) -> dict[str, Any]:
    return {
        "name": name,
        "wake_word": wake_word,
        "sample_duration_s": 2.0,
        "training": dict(DEFAULT_TRAINING),
        "share_token": None,
        "created_at": _now(),
    }


def load_settings(project: Project) -> dict[str, Any]:
    with _lock:
        stored: dict[str, Any] = {}
        if project.file.exists():
            try:
                stored = json.loads(project.file.read_text())
            except json.JSONDecodeError:
                stored = {}
    settings = _default_settings(project.id, stored.get("wake_word", project.id))
    settings.update({k: v for k, v in stored.items() if k != "training"})
    settings["training"].update(stored.get("training", {}))
    return settings


def write_settings(project: Project, settings: dict[str, Any]) -> None:
    with _lock:
        project.dir.mkdir(parents=True, exist_ok=True)
        project.file.write_text(json.dumps(settings, indent=2, ensure_ascii=False))


def save_settings(project: Project, update: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings(project)
    if "wake_word" in update:
        settings["wake_word"] = str(update["wake_word"]).strip() or settings["wake_word"]
    if "name" in update:
        settings["name"] = str(update["name"]).strip() or settings["name"]
    if "sample_duration_s" in update:
        settings["sample_duration_s"] = float(update["sample_duration_s"])
    if "training" in update and isinstance(update["training"], dict):
        for key, default in DEFAULT_TRAINING.items():
            if key in update["training"] and update["training"][key] is not None:
                settings["training"][key] = type(default)(update["training"][key])
    write_settings(project, settings)
    return settings


# ----- project registry -----------------------------------------------------------
def list_projects() -> list[dict[str, Any]]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    current = current_project_id()
    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir() or not PROJECT_ID.match(d.name):
            continue
        p = Project(d.name)
        s = load_settings(p)
        out.append({
            "id": p.id,
            "name": s["name"],
            "wake_word": s["wake_word"],
            "created_at": s.get("created_at"),
            "positive_count": len(list(p.positive_dir.glob("*.wav"))) if p.positive_dir.exists() else 0,
            "negative_count": len(list(p.negative_dir.glob("*.wav"))) if p.negative_dir.exists() else 0,
            "jobs": len([j for j in p.jobs_dir.iterdir() if j.is_dir()]) if p.jobs_dir.exists() else 0,
            "current": p.id == current,
            "shared": bool(s.get("share_token")),
        })
    return out


def get_project(pid: str) -> Project:
    if not PROJECT_ID.match(pid or "") or not (PROJECTS_DIR / pid).is_dir():
        raise KeyError(pid)
    return Project(pid).ensure()


def create_project(name: str, wake_word: str | None = None) -> Project:
    base = slugify(name)[:32] or "wakeword"
    pid = base
    n = 2
    while (PROJECTS_DIR / pid).exists():
        pid = f"{base}-{n}"
        n += 1
    project = Project(pid).ensure()
    write_settings(project, _default_settings(name.strip() or pid, (wake_word or name).strip() or pid))
    return project


def delete_project(pid: str) -> None:
    project = get_project(pid)
    shutil.rmtree(project.dir)
    if current_project_id() == pid:
        remaining = list_projects()
        select_project(remaining[0]["id"] if remaining else create_project("default", "chaloupko").id)


def current_project_id() -> str | None:
    try:
        pid = CURRENT_FILE.read_text().strip()
    except FileNotFoundError:
        return None
    return pid if PROJECT_ID.match(pid) and (PROJECTS_DIR / pid).is_dir() else None


def select_project(pid: str) -> Project:
    project = get_project(pid)
    with _lock:
        CURRENT_FILE.write_text(project.id)
    return project


def current_project() -> Project:
    pid = current_project_id()
    if pid:
        return Project(pid).ensure()
    projects = list_projects()
    if projects:
        return select_project(projects[0]["id"])
    return select_project(create_project("default", "chaloupko").id)


def find_project_by_token(token: str) -> Project | None:
    if not token:
        return None
    for entry in list_projects():
        p = Project(entry["id"])
        stored = load_settings(p).get("share_token")
        if stored and secrets.compare_digest(str(stored), token):
            return p
    return None


def set_share_token(project: Project, enabled: bool) -> str | None:
    settings = load_settings(project)
    settings["share_token"] = secrets.token_urlsafe(18) if enabled else None
    write_settings(project, settings)
    return settings["share_token"]


# ----- legacy layout migration (single project directly in /data) --------------------
def migrate_legacy_layout() -> None:
    legacy_positive = DATA_DIR / "positive_samples"
    legacy_file = DATA_DIR / "project.json"
    if not legacy_positive.exists() and not legacy_file.exists():
        return
    stored: dict[str, Any] = {}
    if legacy_file.exists():
        try:
            stored = json.loads(legacy_file.read_text())
        except json.JSONDecodeError:
            stored = {}
    wake_word = stored.get("wake_word") or "chaloupko"
    project = create_project(wake_word, wake_word)
    for name in ("positive_samples", "negative_samples", "jobs"):
        src = DATA_DIR / name
        dst = project.dir / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    settings = load_settings(project)
    settings.update({k: v for k, v in stored.items() if k in ("wake_word", "sample_duration_s")})
    settings["training"].update(stored.get("training", {}))
    write_settings(project, settings)
    legacy_file.unlink(missing_ok=True)
    select_project(project.id)


for _d in (DATASETS_DIR, FEATURE_CACHE_DIR, PROJECTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
migrate_legacy_layout()
current_project()


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
