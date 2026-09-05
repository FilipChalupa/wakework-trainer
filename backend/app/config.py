"""Paths and persisted project settings."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")).resolve()
POSITIVE_DIR = DATA_DIR / "positive_samples"
NEGATIVE_DIR = DATA_DIR / "negative_samples"
DATASETS_DIR = DATA_DIR / "negative_datasets"
FEATURE_CACHE_DIR = DATA_DIR / "features_cache"
JOBS_DIR = DATA_DIR / "jobs"
PROJECT_FILE = DATA_DIR / "project.json"

for _d in (POSITIVE_DIR, NEGATIVE_DIR, DATASETS_DIR, FEATURE_CACHE_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_TRAINING = {
    "training_steps": 4000,
    "learning_rate": 0.001,
    "batch_size": 128,
    "eval_step_interval": 250,
    "augmentations_per_sample": 40,
    "clip_duration_ms": 1500,
    "negative_class_weight": 10.0,
    "positive_class_weight": 1.0,
}

DEFAULT_PROJECT: dict[str, Any] = {
    "wake_word": "chaloupko",
    "sample_duration_s": 2.0,
    "training": dict(DEFAULT_TRAINING),
}

_lock = threading.Lock()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    text = text.strip("_")
    return text or "wakeword"


def load_project() -> dict[str, Any]:
    with _lock:
        if PROJECT_FILE.exists():
            try:
                stored = json.loads(PROJECT_FILE.read_text())
            except json.JSONDecodeError:
                stored = {}
        else:
            stored = {}
    project = json.loads(json.dumps(DEFAULT_PROJECT))
    project.update({k: v for k, v in stored.items() if k != "training"})
    project["training"].update(stored.get("training", {}))
    return project


def save_project(update: dict[str, Any]) -> dict[str, Any]:
    project = load_project()
    if "wake_word" in update:
        project["wake_word"] = str(update["wake_word"]).strip() or project["wake_word"]
    if "sample_duration_s" in update:
        project["sample_duration_s"] = float(update["sample_duration_s"])
    if "training" in update and isinstance(update["training"], dict):
        for key, default in DEFAULT_TRAINING.items():
            if key in update["training"] and update["training"][key] is not None:
                project["training"][key] = type(default)(update["training"][key])
    with _lock:
        PROJECT_FILE.write_text(json.dumps(project, indent=2, ensure_ascii=False))
    return project
