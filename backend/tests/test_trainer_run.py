import random

import numpy as np

from trainer.audio_data import SR
from trainer.run import make_hard_negatives, parse_roc, summarize_roc


def test_hard_negatives_from_a_word():
    audio = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    variants = make_hard_negatives(audio, random.Random(0))
    assert len(variants) == 3
    assert variants[2].shape[0] == SR  # swapped halves keep the length
    assert variants[0].shape[0] < SR


def test_hard_negatives_skip_too_short_clips():
    assert make_hard_negatives(np.zeros(1000, dtype=np.float32), random.Random(0)) == []


ROC = """AUC 0.12345
Cutoff 0.10: frr=0.0000; faph=3.000
Cutoff 0.40: frr=0.0500; faph=0.500
Cutoff 0.70: frr=0.2000; faph=0.000
"""


def test_parse_and_summarize_roc():
    auc, points = parse_roc(ROC)
    assert auc == 0.12345 and len(points) == 3
    summary, cutoff = summarize_roc(auc, points)
    assert summary["cutoff"] == 0.40  # lowest FRR among points with <= 0.5 FA/h
    assert summary["points"] == points
    assert 0.6 <= cutoff <= 0.97


def test_summarize_roc_without_points():
    assert summarize_roc(None, []) == (None, 0.97)


def test_datasets_registry_and_converter(tmp_path):
    import soundfile as sf

    from app.datasets import DATASETS, convert_audio_tree

    assert DATASETS["mit_rirs"]["required"] and DATASETS["esc50"]["convert"]["max_seconds"] == 5.0
    src = tmp_path / "sub" / "clip.wav"
    src.parent.mkdir()
    sf.write(str(src), np.zeros(44100 * 2, dtype=np.float32), 44100, subtype="PCM_16")
    (tmp_path / "meta.csv").write_text("x")
    count = convert_audio_tree(tmp_path, max_seconds=1.0, max_files=None)
    assert count == 1
    data, sr = sf.read(str(src))
    assert sr == 16000 and abs(len(data) - 16000) < 100
    assert not (tmp_path / "meta.csv").exists()
