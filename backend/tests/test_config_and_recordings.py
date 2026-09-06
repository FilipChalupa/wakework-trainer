import io
import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app import config
from app.main import app


def _wav_bytes(seconds=1.0, amplitude=0.5, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    start, end = int(0.3 * sr), int(0.7 * sr)
    audio[start:end] = amplitude * np.sin(2 * np.pi * 300 * t[start:end])
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


client = TestClient(app)


def test_default_project_exists():
    res = client.get("/api/projects").json()
    assert res["current"] and any(p["current"] for p in res["items"])


def test_create_select_delete_project():
    res = client.post("/api/projects", json={"name": "Hey Jarvis", "wake_word": "hey jarvis"}).json()
    assert res["current"] == "hey_jarvis"
    cfg = client.get("/api/config").json()["project"]
    assert cfg["wake_word"] == "hey jarvis" and cfg["id"] == "hey_jarvis"
    other = [p["id"] for p in res["items"] if p["id"] != "hey_jarvis"][0]
    assert client.post(f"/api/projects/{other}/select").json()["current"] == other
    assert "hey_jarvis" not in [p["id"] for p in client.delete("/api/projects/hey_jarvis").json()["items"]]


def test_upload_analyze_delete_restore():
    up = client.post("/api/recordings", data={"kind": "positive"}, files={"file": ("a.wav", _wav_bytes(), "audio/wav")}).json()
    assert up["duration"] == 1.0 and up["quality"]["issues"] == [] and len(up["peaks"]) == 48
    listing = client.get("/api/recordings?kind=positive").json()
    assert listing["count"] == 1
    assert client.delete(f"/api/recordings/positive/{up['id']}").json()["restorable"]
    assert client.get("/api/recordings?kind=positive").json()["count"] == 0
    assert client.post(f"/api/recordings/positive/{up['id']}/restore").json()["id"] == up["id"]
    client.delete(f"/api/recordings/positive/{up['id']}")


def test_quality_flags_clipping_and_silence():
    clipped = client.post("/api/recordings", data={"kind": "negative"}, files={"file": ("c.wav", _wav_bytes(amplitude=1.2), "audio/wav")}).json()
    assert "clipping" in clipped["quality"]["issues"]
    silent = client.post("/api/recordings", data={"kind": "negative"}, files={"file": ("s.wav", _wav_bytes(amplitude=0.0), "audio/wav")}).json()
    assert "silent" in silent["quality"]["issues"]
    for rec in (clipped, silent):
        client.delete(f"/api/recordings/negative/{rec['id']}")


def test_contributor_link_flow():
    pid = client.get("/api/projects").json()["current"]
    token = client.post(f"/api/projects/{pid}/share", json={"enabled": True}).json()["share_token"]
    assert token
    assert client.get("/api/contribute/info?token=nope").status_code == 403
    info = client.get(f"/api/contribute/info?token={token}").json()
    assert info["wake_word"]
    up = client.post(f"/api/contribute/recordings?token={token}&name=Babi Jana", data={"kind": "positive"}, files={"file": ("a.wav", _wav_bytes(), "audio/wav")}).json()
    assert up["contributor"] == "babi_jana" and "token=" in up["url"]
    own = client.get(f"/api/contribute/recordings?token={token}&name=Babi Jana").json()
    assert own["count"] == 1
    assert client.delete(f"/api/contribute/recordings/positive/{up['id']}?token={token}&name=Someone Else").status_code == 403
    assert client.delete(f"/api/contribute/recordings/positive/{up['id']}?token={token}&name=Babi Jana").status_code == 200
    assert client.post(f"/api/projects/{pid}/share", json={"enabled": False}).json()["share_token"] is None
    assert client.get(f"/api/contribute/info?token={token}").status_code == 403


def test_train_requires_samples():
    res = client.post("/api/train")
    assert res.status_code == 400 and res.json()["detail"]["code"] == "too_few_samples"


def test_legacy_layout_migration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "CURRENT_FILE", tmp_path / "current_project")
    (tmp_path / "positive_samples").mkdir()
    (tmp_path / "positive_samples" / "x.wav").write_bytes(b"x")
    (tmp_path / "project.json").write_text(json.dumps({"wake_word": "hej domku", "training": {"training_steps": 123}}))
    config.migrate_legacy_layout()
    project = config.current_project()
    assert project.id == "hej_domku"
    assert (project.positive_dir / "x.wav").exists()
    assert config.load_settings(project)["training"]["training_steps"] == 123
    assert not (tmp_path / "project.json").exists()
