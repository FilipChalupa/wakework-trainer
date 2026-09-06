"""Streaming wake word detectors used by the live test, the monitor, evaluation and the auto threshold.

StreamingDetector     – microWakeWord streaming .tflite (micro-frontend features, sliding-window average)
OwwStreamingDetector  – openWakeWord-style classifier (speech embeddings), same interface
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from .config import Project, current_project

FRAME_BYTES = 160 * 2  # 10 ms of 16 kHz int16 audio
REFRACTORY_SLICES = 25  # like microWakeWord's ignore_slices_after_accept (~0.75 s)
WARMUP_SLICES = 25  # the streaming model's zero initial state produces a short transient; ignore it (as microWakeWord's tests do)
MONITOR_KEEP_BYTES = 16000 * 2 * 3  # last 3 s of audio kept for saving detections in monitor mode


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


class OwwStreamingDetector:
    """Same interface as StreamingDetector, for openWakeWord-style (Wyoming) classifiers."""

    def __init__(self, model_path: Path, cutoff: float = 0.5, window: int = 1):
        from trainer.oww_features import OwwClassifier, OwwFeatures, OwwStreamer, ensure_oww_models

        from .config import FEATURE_CACHE_DIR

        self.features = _shared_oww_features(ensure_oww_models(FEATURE_CACHE_DIR))
        self.streamer = OwwStreamer(self.features)
        self.classifier = OwwClassifier(model_path)
        self.cutoff = float(cutoff)
        self.window = max(1, int(window))  # here: consecutive frames above the cutoff needed (wyoming "trigger level")
        self._above = 0
        self._refractory = 0
        self._warmup = 20  # ~1.6 s: the streamer's zero-filled buffers settle
        self._buffer = b""
        self.max_average = 0.0
        self.detections = 0

    def recent_audio(self) -> bytes:
        return self.streamer._history[-16000 * 3 :].astype(np.int16).tobytes()

    def feed(self, pcm: bytes) -> list[dict[str, Any]]:
        self._buffer += pcm
        usable = len(self._buffer) - len(self._buffer) % 2
        samples = np.frombuffer(self._buffer[:usable], dtype=np.int16)
        self._buffer = self._buffer[usable:]
        results: list[dict[str, Any]] = []
        from trainer.oww_features import CHUNK_SAMPLES

        frames: list[np.ndarray] = []
        # feed chunk by chunk so every produced embedding frame gets its own prediction (a single big feed
        # would only leave the final window)
        for start in range(0, samples.shape[0], CHUNK_SAMPLES):
            for _ in range(self.streamer.feed(samples[start : start + CHUNK_SAMPLES])):
                frames.append(self.streamer.window().copy())
        for window in frames:
            probability = self.classifier.predict(window)
            detected = False
            if self._warmup > 0:
                self._warmup -= 1
                results.append({"p": round(probability, 4), "avg": round(probability, 4), "detected": False, "warmup": True})
                continue
            self.max_average = max(self.max_average, probability)
            self._above = self._above + 1 if probability >= self.cutoff else 0
            if self._refractory > 0:
                self._refractory -= 1
            elif self._above >= self.window:
                detected = True
                self.detections += 1
                self._refractory = 12  # ~1 s
                self._above = 0
            results.append({"p": round(probability, 4), "avg": round(probability, 4), "detected": detected})
        return results


_oww_features_cache: dict[str, Any] = {}


def _shared_oww_features(models_dir: Path):
    from trainer.oww_features import OwwFeatures

    key = str(models_dir)
    if key not in _oww_features_cache:
        _oww_features_cache[key] = OwwFeatures(models_dir)
    return _oww_features_cache[key]


def make_detector(model_path: Path, cutoff: float, window: int):
    from trainer.oww_features import is_oww_model

    if is_oww_model(model_path):
        return OwwStreamingDetector(model_path, cutoff=cutoff, window=window)
    return StreamingDetector(model_path, cutoff=cutoff, window=window)


def _job_dir(job_id: str) -> Path:
    from .jobs import find_job_dir

    return find_job_dir(job_id)


def _model_for_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    job_dir = _job_dir(job_id)
    job = json.loads((job_dir / "job.json").read_text())
    model = job_dir / f"{job['slug']}.tflite"
    if not model.exists():
        raise HTTPException(404, {"code": "not_found", "message": "Model not found"})
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
    detector = make_detector(model_path, cutoff=cutoff, window=window)
    # leading silence covers the model's start-up transient, trailing silence lets it see the end of the word
    detector.feed(bytes(FRAME_BYTES * 90) + pcm + bytes(FRAME_BYTES * 40))
    return {"max_probability": round(detector.max_average, 4), "detections": detector.detections}


