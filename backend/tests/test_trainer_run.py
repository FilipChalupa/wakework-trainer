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


def test_version_and_update_check(monkeypatch):
    from app import system

    monkeypatch.setenv("APP_VERSION", "1.1.0")
    assert system.app_version() == "1.1.0"
    system._update_cache.update(at=0.0, value=None)

    class R:
        ok = True

        @staticmethod
        def json():
            return [{"name": "v1.0.0"}, {"name": "v1.2.0"}, {"name": "junk"}]

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    system._cache.update(at=0.0, value=None)
    info = system.system_info()
    assert info["version"] == "1.1.0" and info["latest_version"] == "v1.2.0" and info["update_available"] is True


def test_pick_best_window_is_json_serializable_and_prefers_separation():
    import json

    from trainer.run import pick_best_window

    cands = [
        {"window": 3, "margin": 0.1, "cutoff": 0.6, "positives_passed": 9, "negatives_triggered": 0},
        {"window": 5, "margin": 0.2, "cutoff": 0.7, "positives_passed": 10, "negatives_triggered": 0},
        {"window": 7, "margin": 0.25, "cutoff": 0.8, "positives_passed": 8, "negatives_triggered": 0},
    ]
    best = pick_best_window(cands)
    assert best["window"] == 5 and len(best["candidates"]) == 3
    json.dumps(best)  # must not raise (no circular reference)
