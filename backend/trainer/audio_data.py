"""Clip loading, silence trimming, synthetic noise and ambient set generation."""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

SR = 16000


def load_wav(path: Path | str) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def save_wav(path: Path | str, audio: np.ndarray) -> None:
    sf.write(str(path), np.clip(audio, -1.0, 1.0), SR, subtype="PCM_16")


def peak_normalize(audio: np.ndarray, target: float = 0.7) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-5:
        return audio
    return (audio * (target / peak)).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold_db: float = -32.0, pad_ms: int = 120, frame: int = 160) -> np.ndarray:
    """Energy based trimming of leading/trailing silence relative to the loudest frame."""
    if audio.shape[0] < frame * 4:
        return audio
    usable = audio[: (audio.shape[0] // frame) * frame].reshape(-1, frame)
    rms = np.sqrt((usable ** 2).mean(axis=1) + 1e-12)
    peak = rms.max()
    if peak < 1e-4:
        return audio
    threshold = peak * (10 ** (threshold_db / 20.0))
    active = np.where(rms > threshold)[0]
    if active.size == 0:
        return audio
    pad = int(SR * pad_ms / 1000)
    start = max(0, int(active[0]) * frame - pad)
    end = min(audio.shape[0], (int(active[-1]) + 1) * frame + pad)
    return audio[start:end]


def stable_split(paths: list[Path], validation: float = 0.1, test: float = 0.1) -> dict[str, list[Path]]:
    """Deterministic split by filename hash. Guarantees at least one clip per split when possible."""
    train, val, tst = [], [], []
    for p in sorted(paths):
        h = int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 1000 / 1000.0
        if h < test:
            tst.append(p)
        elif h < test + validation:
            val.append(p)
        else:
            train.append(p)
    if len(paths) >= 3:
        if not val:
            val.append(train.pop())
        if not tst:
            tst.append(train.pop())
    if not train:
        train = list(paths)
    if not val:
        val = list(train)
    if not tst:
        tst = list(train)
    return {"train": train, "validation": val, "test": tst}


class WavClips:
    """Minimal drop-in replacement for microwakeword.audio.clips.Clips backed by soundfile."""

    def __init__(self, splits: dict[str, list[Path]], trim: bool = False, normalize: bool = True, cache: bool = True):
        self.splits = splits
        self.trim = trim
        self.normalize = normalize
        self.use_cache = cache
        self.clips = sorted({p for v in splits.values() for p in v})
        self._cache: dict[Path, np.ndarray] = {}

    def _load(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is None:
            audio = load_wav(path)
            if self.normalize:
                audio = peak_normalize(audio)
            if self.trim:
                audio = trim_silence(audio)
            cached = audio
            if self.use_cache:
                self._cache[path] = cached
        return cached

    def durations(self, split: str = "train") -> list[float]:
        return [self._load(p).shape[0] / SR for p in self.splits[split]]

    def audio_generator(self, split: str | None = None, repeat: int = 1):
        paths = self.clips if split is None else self.splits[split]
        for _ in range(repeat):
            for p in paths:
                yield self._load(p)

    def get_random_clip(self) -> np.ndarray:
        return self._load(random.choice(self.splits["train"]))

    def random_audio_generator(self, max_clips: int = math.inf):
        while max_clips > 0:
            max_clips -= 1
            yield self.get_random_clip()


# ----- synthetic noise --------------------------------------------------------
def _colored_noise(n: int, color: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    if color == "white":
        out = white
    elif color == "pink":
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        out = lfilter(b, a, white).astype(np.float32)
    elif color == "brown":
        out = np.cumsum(white).astype(np.float32)
        out -= np.linspace(out[0], out[-1], n, dtype=np.float32)
    elif color == "hum":
        t = np.arange(n) / SR
        out = (np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        out += 0.05 * white
    else:
        out = white
    return peak_normalize(out, 0.5)


def ensure_synthetic_noise(noise_dir: Path, seconds: float = 8.0, seed: int = 1234) -> list[Path]:
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    created = []
    for color in ("white", "pink", "brown", "hum"):
        for variant in range(2):
            path = noise_dir / f"{color}_{variant}.wav"
            if not path.exists():
                audio = _colored_noise(n, color, rng)
                envelope = 0.4 + 0.6 * np.abs(np.sin(np.linspace(0, 3 + variant, n))).astype(np.float32)
                save_wav(path, audio * envelope)
            created.append(path)
    # near silence with a bit of hiss
    path = noise_dir / "silence_0.wav"
    if not path.exists():
        save_wav(path, rng.standard_normal(n).astype(np.float32) * 0.002)
    created.append(path)
    return created


def build_ambient_clip(sources: list[Path], seconds: float, rng: random.Random, noise: list[Path] | None = None) -> np.ndarray:
    """Concatenates random negative clips with random gaps/gains into one long clip."""
    total = int(seconds * SR)
    out = np.zeros(total, dtype=np.float32)
    pos = int(rng.uniform(0, SR * 0.5))
    while pos < total and sources:
        clip = load_wav(rng.choice(sources))
        clip = trim_silence(clip) if clip.shape[0] > SR // 2 else clip
        gain = 10 ** (rng.uniform(-18, 0) / 20.0)
        clip = peak_normalize(clip, 0.8) * gain
        end = min(total, pos + clip.shape[0])
        out[pos:end] += clip[: end - pos]
        pos = end + int(rng.uniform(0.05, 0.8) * SR)
    if noise:
        bed = np.zeros(total, dtype=np.float32)
        p = 0
        while p < total:
            n = load_wav(rng.choice(noise))
            e = min(total, p + n.shape[0])
            bed[p:e] = n[: e - p]
            p = e
        out += bed * (10 ** (rng.uniform(-35, -18) / 20.0))
    return np.clip(out, -1.0, 1.0)
