"""WAV normalisation: whatever the browser sends becomes 16 kHz / mono / 16-bit PCM."""
from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def _ffmpeg_convert(raw: bytes) -> bytes | None:
    if shutil.which("ffmpeg") is None:
        return None
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
            "-f", "wav", "pipe:1",
        ],
        input=raw,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def _python_convert(raw: bytes) -> bytes:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd

        g = gcd(sr, TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, sr // g).astype(np.float32)
    mono = np.clip(mono, -1.0, 1.0)
    out = io.BytesIO()
    sf.write(out, mono, TARGET_SR, subtype="PCM_16", format="WAV")
    return out.getvalue()


def normalize_wav(raw: bytes) -> tuple[bytes, float]:
    """Returns (wav_bytes, duration_seconds)."""
    converted = _ffmpeg_convert(raw)
    if converted is None:
        converted = _python_convert(raw)
    info = sf.info(io.BytesIO(converted))
    if info.samplerate != TARGET_SR or info.channels != 1 or info.subtype != "PCM_16":
        converted = _python_convert(converted)
        info = sf.info(io.BytesIO(converted))
    return converted, float(info.frames) / info.samplerate


def wav_duration(path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / info.samplerate
    except Exception:
        return 0.0
