"""Live wake word testing: streaming TFLite inference over microphone audio (WebSocket) and
offline evaluation of a model on the stored recordings."""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

import threading

from .config import Project, current_project

router = APIRouter(prefix="/api", tags=["test"])
MONITOR_KEEP_BYTES = 16000 * 2 * 3  # last 3 s of audio kept for saving detections
MONITOR_MAX_FILES = 200

_eval_lock = threading.Semaphore(1)  # one offline evaluation at a time
_live_sessions = threading.Semaphore(2)  # at most two concurrent live tests

FRAME_BYTES = 160 * 2  # 10 ms of 16 kHz int16 audio
REFRACTORY_SLICES = 25  # like microWakeWord's ignore_slices_after_accept (~0.75 s)
WARMUP_SLICES = 25  # the streaming model's zero initial state produces a short transient; ignore it (as microWakeWord's tests do)


class StreamingDetector:
    """Feeds raw 16 kHz int16 PCM through the micro-frontend and a streaming microWakeWord model."""

    def __init__(self, model_path: Path, cutoff: float = 0.97, window: int = 5):
        from microwakeword.inference import Model
        from pymicro_features import MicroFrontend

        self.model = Model(str(model_path))
        self.frontend = MicroFrontend()
        self.slices = int(self.model.input_feature_slices)
        self.cutoff = float(cutoff)
        self.window = max(1, int(window))
        self._buffer = b""
        self._history = b""  # last few seconds of PCM (for saving detections in monitor mode)
        self._frames: list[list[float]] = []
        self._recent: deque[float] = deque(maxlen=self.window)
        self._refractory = 0
        self._warmup = WARMUP_SLICES
        self.max_average = 0.0
        self.detections = 0

    def recent_audio(self) -> bytes:
        return self._history[-MONITOR_KEEP_BYTES:]

    def feed(self, pcm: bytes) -> list[dict[str, Any]]:
        self._history = (self._history + pcm)[-MONITOR_KEEP_BYTES:]
        self._buffer += pcm
        results: list[dict[str, Any]] = []
        idx = 0
        n = len(self._buffer)
        while idx + FRAME_BYTES <= n:
            out = self.frontend.process_samples(self._buffer[idx : idx + FRAME_BYTES])
            idx += max(1, out.samples_read) * 2
            if out.features:
                self._frames.append(list(out.features))
            if len(self._frames) >= self.slices:
                chunk = np.asarray(self._frames[: self.slices], dtype=np.float32)
                self._frames = self._frames[self.slices :]
                probability = float(self.model.predict_spectrogram(chunk)[0])
                self._recent.append(probability)
                average = float(sum(self._recent) / len(self._recent))
                detected = False
                if self._warmup > 0:
                    self._warmup -= 1
                    results.append({"p": round(probability, 4), "avg": round(average, 4), "detected": False, "warmup": True})
                    continue
                self.max_average = max(self.max_average, average)
                if self._refractory > 0:
                    self._refractory -= 1
                elif average >= self.cutoff and len(self._recent) == self.window:
                    detected = True
                    self.detections += 1
                    self._refractory = REFRACTORY_SLICES
                results.append({"p": round(probability, 4), "avg": round(average, 4), "detected": detected})
        self._buffer = self._buffer[idx:]
        return results


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


def _job_dir(job_id: str) -> Path:
    from .training import find_job_dir

    return find_job_dir(job_id)


def _model_for_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    job_dir = _job_dir(job_id)
    job = json.loads((job_dir / "job.json").read_text())
    model = job_dir / f"{job['slug']}.tflite"
    if not model.exists():
        raise HTTPException(404, "Model not found")
    manifest_path = job_dir / f"{job['slug']}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return model, manifest


def _load_pcm16(path: Path) -> bytes:
    audio, sr = sf.read(str(path), dtype="int16", always_2d=True)
    mono = audio[:, 0] if audio.shape[1] == 1 else audio.mean(axis=1).astype(np.int16)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, 16000)
        mono = np.clip(resample_poly(mono.astype(np.float32), 16000 // g, sr // g), -32768, 32767).astype(np.int16)
    return mono.tobytes()


def evaluate_clip(model_path: Path, pcm: bytes, cutoff: float, window: int) -> dict[str, Any]:
    detector = StreamingDetector(model_path, cutoff=cutoff, window=window)
    # leading silence covers the model's start-up transient, trailing silence lets it see the end of the word
    detector.feed(bytes(FRAME_BYTES * 90) + pcm + bytes(FRAME_BYTES * 40))
    return {"max_probability": round(detector.max_average, 4), "detections": detector.detections}


@router.get("/jobs/{job_id}/test-info")
def test_info(job_id: str):
    model, manifest = _model_for_job(job_id)
    micro = manifest.get("micro", {})
    return {
        "job_id": job_id,
        "model": model.name,
        "probability_cutoff": micro.get("probability_cutoff", 0.97),
        "sliding_window_size": micro.get("sliding_window_size", 5),
    }


@router.post("/jobs/{job_id}/evaluate")
async def evaluate_job(job_id: str, cutoff: float | None = None, window: int = 5):
    model, manifest = _model_for_job(job_id)
    if cutoff is None:
        cutoff = float(manifest.get("micro", {}).get("probability_cutoff", 0.97))
    job = json.loads((_job_dir(job_id) / "job.json").read_text())
    project = Project(job["project_id"]) if job.get("project_id") else current_project()
    if not _eval_lock.acquire(blocking=False):
        raise HTTPException(429, {"code": "busy", "message": "Another evaluation is running"})

    def run() -> dict[str, Any]:
        from .recordings import contributor_of, read_meta

        results = []
        for kind, folder in (("positive", project.positive_dir), ("negative", project.negative_dir)):
            meta = read_meta(kind, project)
            for path in sorted(folder.glob("*.wav")):
                try:
                    res = evaluate_clip(model, _load_pcm16(path), cutoff, window)
                except Exception as exc:  # noqa: BLE001
                    res = {"max_probability": None, "detections": 0, "error": str(exc)}
                results.append({
                    "id": path.name,
                    "kind": kind,
                    "url": f"/api/recordings/{kind}/{path.name}",
                    "tag": meta.get(path.name, {}).get("tag"),
                    "contributor": contributor_of(path.name),
                    **res,
                })
        positives = [r for r in results if r["kind"] == "positive" and r["max_probability"] is not None]
        negatives = [r for r in results if r["kind"] == "negative" and r["max_probability"] is not None]
        # outliers: wake word recordings the model scores far below the typical one (likely bad takes),
        # negatives that trigger the model (likely contain the wake word or a look-alike)
        median = float(np.median([r["max_probability"] for r in positives])) if positives else 0.0
        for r in results:
            if r["max_probability"] is None:
                r["outlier"] = False
            elif r["kind"] == "positive":
                r["outlier"] = len(positives) >= 4 and r["max_probability"] < max(0.5 * median, median - 0.3)
            else:
                r["outlier"] = r["detections"] > 0
        by_tag: dict[str, dict[str, int]] = {}
        for r in positives:
            entry = by_tag.setdefault(r["tag"] or "normal", {"total": 0, "detected": 0})
            entry["total"] += 1
            entry["detected"] += 1 if r["detections"] > 0 else 0
        return {
            "cutoff": cutoff,
            "window": window,
            "items": results,
            "summary": {
                "positive_total": len(positives),
                "positive_detected": sum(1 for r in positives if r["detections"] > 0),
                "negative_total": len(negatives),
                "negative_triggered": sum(1 for r in negatives if r["detections"] > 0),
                "by_tag": by_tag,
                "outliers": sum(1 for r in results if r.get("outlier")),
                "median_positive": round(median, 3),
            },
        }

    try:
        return await run_in_threadpool(run)
    finally:
        _eval_lock.release()


@router.websocket("/test/ws")
async def test_websocket(websocket: WebSocket):
    await websocket.accept()
    params = websocket.query_params
    job_id = params.get("job_id", "")
    try:
        model, manifest = _model_for_job(job_id)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "message": str(exc.detail)})
        await websocket.close()
        return
    micro = manifest.get("micro", {})
    cutoff = float(params.get("cutoff") or micro.get("probability_cutoff", 0.97))
    window = int(params.get("window") or micro.get("sliding_window_size", 5))
    save = params.get("save") in ("1", "true", "yes")
    job = json.loads((_job_dir(job_id) / "job.json").read_text())
    project = Project(job["project_id"]) if job.get("project_id") else current_project()
    if not _live_sessions.acquire(blocking=False):
        await websocket.send_json({"type": "error", "message": "Too many live test sessions"})
        await websocket.close()
        return
    try:
        detector = await run_in_threadpool(StreamingDetector, model, cutoff, window)
    except Exception as exc:  # noqa: BLE001
        _live_sessions.release()
        await websocket.send_json({"type": "error", "message": f"Could not load model: {exc}"})
        await websocket.close()
        return
    await websocket.send_json({"type": "ready", "cutoff": cutoff, "window": window, "model": model.name})
    started = time.time()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is None:
                text = message.get("text") or ""
                if text == "reset":
                    detector = await run_in_threadpool(StreamingDetector, model, cutoff, window)
                    await websocket.send_json({"type": "ready", "cutoff": cutoff, "window": window, "model": model.name})
                continue
            results = await run_in_threadpool(detector.feed, data)
            if results:
                await websocket.send_json({
                    "type": "frames",
                    "t": round(time.time() - started, 3),
                    "frames": results,
                    "detections": detector.detections,
                    "max": round(detector.max_average, 4),
                })
                if save and any(f["detected"] for f in results):
                    try:
                        item = await run_in_threadpool(save_detection, project, detector.recent_audio(), job_id)
                        await websocket.send_json({"type": "detection", "item": item, "t": round(time.time() - started, 3)})
                    except Exception as exc:  # noqa: BLE001
                        await websocket.send_json({"type": "error", "message": f"Could not save detection: {exc}"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        _live_sessions.release()


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
        raise HTTPException(400, "Bad name")
    path = monitor_dir(current_project()) / name
    if not path.exists():
        raise HTTPException(404, "Not found")
    from fastapi.responses import FileResponse

    return FileResponse(path, media_type="audio/wav", filename=name)


@router.post("/monitor/{name}/negative")
def monitor_to_negative(name: str):
    """Moves a saved false activation into the project's negative samples (tagged 'noisy' when it came from a monitor)."""
    from .recordings import describe, set_tag

    if not SAFE_NAME.match(name):
        raise HTTPException(400, "Bad name")
    project = current_project()
    src = monitor_dir(project) / name
    if not src.exists():
        raise HTTPException(404, "Not found")
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
        raise HTTPException(400, "Bad name")
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
