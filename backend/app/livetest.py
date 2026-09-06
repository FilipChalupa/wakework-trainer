"""Live wake word testing: streaming inference over microphone audio (WebSocket) and offline evaluation
of a model on the stored recordings. Detectors live in detectors.py, saved activations in monitor.py."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .config import Project, current_project
from .detectors import _job_dir, _load_pcm16, _model_for_job, evaluate_clip, make_detector
from .monitor import save_detection

router = APIRouter(prefix="/api", tags=["test"])

_eval_lock = threading.Semaphore(1)  # one offline evaluation at a time
_live_sessions = threading.Semaphore(2)  # at most two concurrent live tests



@router.get("/jobs/{job_id}/test-info")
def test_info(job_id: str):
    model, manifest = _model_for_job(job_id)
    micro = manifest.get("micro", {})
    job = json.loads((_job_dir(job_id) / "job.json").read_text())
    target = (job.get("training") or {}).get("target", "esphome")
    return {
        "job_id": job_id,
        "model": model.name,
        "target": target,
        "probability_cutoff": micro.get("probability_cutoff", 0.5 if target == "wyoming" else 0.97),
        "sliding_window_size": micro.get("sliding_window_size", 1 if target == "wyoming" else 5),
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
        detector = await run_in_threadpool(make_detector, model, cutoff, window)
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
                    detector = await run_in_threadpool(make_detector, model, cutoff, window)
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


