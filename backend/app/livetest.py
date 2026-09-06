"""Live wake word testing: streaming TFLite inference over microphone audio (WebSocket) and
offline evaluation of a model on the stored recordings."""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .config import JOBS_DIR, NEGATIVE_DIR, POSITIVE_DIR

router = APIRouter(prefix="/api", tags=["test"])

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
        self._frames: list[list[float]] = []
        self._recent: deque[float] = deque(maxlen=self.window)
        self._refractory = 0
        self._warmup = WARMUP_SLICES
        self.max_average = 0.0
        self.detections = 0

    def feed(self, pcm: bytes) -> list[dict[str, Any]]:
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


def _job_dir(job_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_\-]+$", job_id):
        raise HTTPException(400, "Bad job id")
    path = JOBS_DIR / job_id
    if not path.is_dir():
        raise HTTPException(404, "Job not found")
    return path


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

    def run() -> dict[str, Any]:
        results = []
        for kind, folder in (("positive", POSITIVE_DIR), ("negative", NEGATIVE_DIR)):
            for path in sorted(folder.glob("*.wav")):
                try:
                    res = evaluate_clip(model, _load_pcm16(path), cutoff, window)
                except Exception as exc:  # noqa: BLE001
                    res = {"max_probability": None, "detections": 0, "error": str(exc)}
                results.append({"id": path.name, "kind": kind, "url": f"/api/recordings/{kind}/{path.name}", **res})
        positives = [r for r in results if r["kind"] == "positive" and r["max_probability"] is not None]
        negatives = [r for r in results if r["kind"] == "negative" and r["max_probability"] is not None]
        return {
            "cutoff": cutoff,
            "window": window,
            "items": results,
            "summary": {
                "positive_total": len(positives),
                "positive_detected": sum(1 for r in positives if r["detections"] > 0),
                "negative_total": len(negatives),
                "negative_triggered": sum(1 for r in negatives if r["detections"] > 0),
            },
        }

    return await run_in_threadpool(run)


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
    try:
        detector = await run_in_threadpool(StreamingDetector, model, cutoff, window)
    except Exception as exc:  # noqa: BLE001
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
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001
            pass
