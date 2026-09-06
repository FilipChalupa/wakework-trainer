import random

import numpy as np

from trainer.audio_data import SR, build_ambient_clip, ensure_synthetic_noise, peak_normalize, stable_split, trim_silence


def _word(seconds=0.6, gap=0.5):
    t = np.arange(int(seconds * SR)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    silence = np.zeros(int(gap * SR), dtype=np.float32)
    return np.concatenate([silence, tone, silence])


def test_trim_silence_keeps_the_word_with_padding():
    clip = _word()
    trimmed = trim_silence(clip)
    assert 0.6 * SR <= trimmed.shape[0] <= (0.6 + 0.3) * SR
    assert np.abs(trimmed).max() > 0.4


def test_trim_silence_leaves_silence_untouched():
    silence = np.zeros(SR, dtype=np.float32)
    assert trim_silence(silence).shape[0] == SR


def test_peak_normalize():
    out = peak_normalize(np.array([0.1, -0.2, 0.05], dtype=np.float32), target=0.7)
    assert abs(np.abs(out).max() - 0.7) < 1e-6


def test_stable_split_is_deterministic_and_covers_all(tmp_path):
    paths = [tmp_path / f"{i:03d}.wav" for i in range(50)]
    a = stable_split(paths)
    b = stable_split(list(reversed(paths)))
    assert a == b
    assert len(a["train"]) + len(a["validation"]) + len(a["test"]) == 50
    assert a["validation"] and a["test"]


def test_stable_split_tiny_sets_guarantee_every_split(tmp_path):
    paths = [tmp_path / f"{i}.wav" for i in range(3)]
    s = stable_split(paths)
    assert s["train"] and s["validation"] and s["test"]


def test_synthetic_noise_and_ambient(tmp_path):
    files = ensure_synthetic_noise(tmp_path / "noise", seconds=1.0)
    assert len(files) == 9 and all(f.exists() for f in files)
    clip = build_ambient_clip(files, seconds=3.0, rng=random.Random(1), noise=files)
    assert clip.shape[0] == 3 * SR
    assert np.abs(clip).max() <= 1.0


def test_trim_edges_keeps_context_and_ignores_short_silence():
    import io

    import soundfile as sf

    from app.audio import trim_edges

    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    long = np.concatenate([np.zeros(int(1.5 * sr), np.float32), tone, np.zeros(int(1.2 * sr), np.float32)])
    buf = io.BytesIO()
    sf.write(buf, long, sr, subtype="PCM_16", format="WAV")
    out, duration = trim_edges(buf.getvalue())
    assert 0.9 <= duration <= 1.1  # 0.5 s tone + 0.25 s context on both sides
    short = np.concatenate([np.zeros(int(0.2 * sr), np.float32), tone, np.zeros(int(0.2 * sr), np.float32)])
    buf = io.BytesIO()
    sf.write(buf, short, sr, subtype="PCM_16", format="WAV")
    out2, duration2 = trim_edges(buf.getvalue())
    assert out2 == buf.getvalue() and abs(duration2 - 0.9) < 0.01
