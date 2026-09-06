"""openWakeWord feature pipeline re-implemented on top of the TFLite runtime already in the image.

openWakeWord models classify windows of 16 "speech embedding" frames (96 values each). The embeddings come
from Google's speech_embedding model applied to 76-frame (760 ms) mel-spectrogram windows with an 8-frame
(80 ms) step; the mel-spectrogram model consumes 16 kHz int16 audio. Both models are the official ones shipped
with openWakeWord (downloaded once from its GitHub release). Everything here mirrors openwakeword.utils.AudioFeatures,
including the streaming variant used by wyoming-openwakeword, so a model trained here behaves the same on the satellite.
"""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np

OWW_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
OWW_MODEL_FILES = ("melspectrogram.tflite", "embedding_model.tflite")
MEL_BINS = 32
MEL_WINDOW = 76  # mel frames per embedding window (760 ms)
MEL_STEP = 8  # mel frames between embeddings (80 ms)
EMBEDDING_DIM = 96
CLASSIFIER_FRAMES = 16  # embedding frames per classifier input (~2 s of context)
CHUNK_SAMPLES = 1280  # 80 ms of 16 kHz audio = one new embedding frame in streaming mode
SR = 16000


def ensure_oww_models(cache_dir: Path) -> Path:
    """Downloads the melspectrogram + embedding models (about 2.4 MB) into cache_dir/oww_models."""
    folder = Path(cache_dir) / "oww_models"
    folder.mkdir(parents=True, exist_ok=True)
    missing = [f for f in OWW_MODEL_FILES if not (folder / f).exists() or (folder / f).stat().st_size < 100_000]
    if missing:
        import requests

        for name in missing:
            res = requests.get(OWW_RELEASE + name, timeout=120)
            res.raise_for_status()
            (folder / name).write_bytes(res.content)
    return folder


def mel_transform(spec: np.ndarray) -> np.ndarray:
    """openWakeWord scales the model's mel output to match the original TensorFlow implementation."""
    return spec / 10.0 + 2.0


def embedding_windows(n_mel_frames: int) -> int:
    """Number of embedding frames produced from n mel frames."""
    return 0 if n_mel_frames < MEL_WINDOW else (n_mel_frames - MEL_WINDOW) // MEL_STEP + 1


def classifier_windows(n_embedding_frames: int, step: int = 1) -> list[tuple[int, int]]:
    """(start, end) slices of CLASSIFIER_FRAMES embedding frames over a longer embedding sequence."""
    if n_embedding_frames < CLASSIFIER_FRAMES:
        return []
    return [(i, i + CLASSIFIER_FRAMES) for i in range(0, n_embedding_frames - CLASSIFIER_FRAMES + 1, step)]


class OwwFeatures:
    """Batch feature extraction: 16 kHz int16 audio -> (n, 96) speech embeddings."""

    def __init__(self, models_dir: Path, threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        self._lock = threading.Lock()
        self.mel = Interpreter(model_path=str(Path(models_dir) / "melspectrogram.tflite"), num_threads=threads)
        self.emb = Interpreter(model_path=str(Path(models_dir) / "embedding_model.tflite"), num_threads=threads)
        self._mel_size = None
        self._emb_batch = None
        self._mel_in = self.mel.get_input_details()[0]["index"]
        self._mel_out = self.mel.get_output_details()[0]["index"]
        self._emb_in = self.emb.get_input_details()[0]["index"]
        self._emb_out = self.emb.get_output_details()[0]["index"]

    def melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """(samples,) int16/float -> (frames, 32) transformed mel-spectrogram."""
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio * 32767, -32768, 32767)
        x = audio.astype(np.float32)[None, :]
        with self._lock:
            if self._mel_size != x.shape[1]:
                self.mel.resize_tensor_input(0, [1, x.shape[1]], strict=True)
                self.mel.allocate_tensors()
                self._mel_size = x.shape[1]
            self.mel.set_tensor(self._mel_in, x)
            self.mel.invoke()
            spec = np.squeeze(self.mel.get_tensor(self._mel_out))
        return mel_transform(spec.reshape(-1, MEL_BINS))

    def embed_melspec(self, spec: np.ndarray) -> np.ndarray:
        """(frames, 32) mel -> (n_embeddings, 96)."""
        windows = [spec[i : i + MEL_WINDOW] for i in range(0, spec.shape[0] - MEL_WINDOW + 1, MEL_STEP)]
        if not windows:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        batch = np.asarray(windows, dtype=np.float32)[..., None]
        with self._lock:
            if self._emb_batch != batch.shape[0]:
                self.emb.resize_tensor_input(0, [batch.shape[0], MEL_WINDOW, MEL_BINS, 1], strict=True)
                self.emb.allocate_tensors()
                self._emb_batch = batch.shape[0]
            self.emb.set_tensor(self._emb_in, batch)
            self.emb.invoke()
            out = self.emb.get_tensor(self._emb_out)
        return out.reshape(batch.shape[0], EMBEDDING_DIM).astype(np.float32)

    def embed_clip(self, audio: np.ndarray) -> np.ndarray:
        return self.embed_melspec(self.melspectrogram(audio))


class OwwStreamer:
    """Streaming embeddings as wyoming-openwakeword computes them: every 1280 samples one new embedding frame."""

    def __init__(self, features: OwwFeatures):
        self.features = features
        self.raw: deque[np.ndarray] = deque()
        self.raw_len = 0
        self.mel = np.ones((MEL_WINDOW, MEL_BINS), dtype=np.float32)
        self.embeddings = np.zeros((CLASSIFIER_FRAMES, EMBEDDING_DIM), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.int16)
        self._history = np.zeros(0, dtype=np.int16)

    def feed(self, pcm: np.ndarray) -> int:
        """Adds int16 samples; returns how many new embedding frames were produced."""
        self._pending = np.concatenate([self._pending, pcm.astype(np.int16)])
        produced = 0
        while self._pending.shape[0] >= CHUNK_SAMPLES:
            chunk = self._pending[:CHUNK_SAMPLES]
            self._pending = self._pending[CHUNK_SAMPLES:]
            self._history = np.concatenate([self._history, chunk])[-SR * 4 :]
            # the mel model needs 3 extra frames (480 samples) of context in front of the chunk to yield exactly 8 new frames
            need = CHUNK_SAMPLES + 480
            context = self._history[-need:]
            if context.shape[0] < need:
                context = np.concatenate([np.zeros(need - context.shape[0], dtype=np.int16), context])
            new_mel = self.features.melspectrogram(context)[-MEL_STEP:]
            self.mel = np.vstack([self.mel, new_mel])[-(MEL_WINDOW + MEL_STEP * 16) :]
            emb = self.features.embed_melspec(self.mel[-MEL_WINDOW:])
            self.embeddings = np.vstack([self.embeddings, emb])[-CLASSIFIER_FRAMES:]
            produced += 1
        return produced

    def window(self) -> np.ndarray:
        return self.embeddings[-CLASSIFIER_FRAMES:].astype(np.float32)


class OwwClassifier:
    """Runs an openWakeWord-style classifier .tflite on (1, 16, 96) embedding windows."""

    def __init__(self, model_path: Path, threads: int = 1):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]["index"]

    def predict(self, window: np.ndarray) -> float:
        x = window.reshape(1, CLASSIFIER_FRAMES, EMBEDDING_DIM).astype(np.float32)
        self.interpreter.set_tensor(self._in["index"], x)
        self.interpreter.invoke()
        return float(np.squeeze(self.interpreter.get_tensor(self._out)))


def is_oww_model(model_path: Path) -> bool:
    """openWakeWord classifiers take (1, 16, 96) float32; microWakeWord streaming models take (1, 3, 40) int8."""
    try:
        from ai_edge_litert.interpreter import Interpreter

        details = Interpreter(model_path=str(model_path)).get_input_details()[0]
        shape = [int(v) for v in details["shape"]]
        return len(shape) == 3 and shape[1] == CLASSIFIER_FRAMES and shape[2] == EMBEDDING_DIM
    except Exception:  # noqa: BLE001
        return False
