import json

from fastapi.testclient import TestClient

from app.main import app
from app.public import version_tuple

client = TestClient(app)


def _fake_done_job(project_dir, job_id="20260101_000000_fake", slug="chaloupko"):
    job_dir = project_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps({"job_id": job_id, "project_id": project_dir.name, "wake_word": "chaloupko", "slug": slug, "training": {"training_steps": 10}, "created_at": "2026-01-01T00:00:00"}))
    (job_dir / "result.json").write_text(json.dumps({"status": "done", "final_metrics": {"manifest_cutoff": 0.7}}))
    (job_dir / f"{slug}.tflite").write_bytes(b"TFL3fake")
    (job_dir / f"{slug}.json").write_text(json.dumps({"type": "micro", "wake_word": "chaloupko", "model": f"{slug}.tflite", "micro": {"probability_cutoff": 0.7, "sliding_window_size": 5}}))
    return job_dir


def test_public_manifest_model_and_bundle():
    from app import config

    project = config.current_project()
    _fake_done_job(project.dir)
    urls = client.get(f"/api/projects/{project.id}/public-urls").json()
    assert urls["has_model"] and urls["manifest_url"].endswith("/manifest.json") and "micro_wake_word" in urls["snippet"]
    token = urls["token"]
    manifest = client.get(f"/api/public/{token}/manifest.json").json()
    assert manifest["model"].endswith(f"/api/public/{token}/model.tflite")
    assert client.get(f"/api/public/{token}/model.tflite").content == b"TFL3fake"
    assert client.get("/api/public/nope/manifest.json").status_code == 403
    res = client.get("/api/bundle")
    assert res.status_code == 200 and res.headers["content-type"] == "application/zip"
    client.delete("/api/jobs/20260101_000000_fake")


def test_device_events_and_version_check():
    from app import config

    project = config.current_project()
    token = config.get_or_create_model_token(project)
    assert client.post(f"/api/public/{token}/device-event", json={"device": "kitchen", "wake_word": "chaloupko", "esphome_version": "2024.6.0"}).status_code == 200
    events = client.get("/api/monitor/device-events").json()
    assert events["items"][-1]["device"] == "kitchen"
    assert "2024.6.0" in events["outdated_versions"]
    assert version_tuple("2025.1.0") > version_tuple("2024.7.0")


def test_sweep_validation_and_queue_drop():
    assert client.post("/api/train/sweep", json={"param": "hard_negatives", "values": [1]}).status_code == 400
    assert client.post("/api/train/sweep", json={"param": "training_steps", "values": []}).status_code == 400
    assert client.get("/api/train/queue").json()["queue"] == []
    assert client.delete("/api/train/queue/does-not-exist").json()["queue"] == []


def test_monitor_endpoints_and_review_flag(tmp_path):
    import io

    import numpy as np
    import soundfile as sf

    assert client.get("/api/monitor").json()["items"] == []
    assert client.delete("/api/monitor").json()["deleted"] == 0
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16", format="WAV")
    up = client.post("/api/recordings", data={"kind": "negative"}, files={"file": ("n.wav", buf.getvalue(), "audio/wav")}).json()
    flagged = client.put(f"/api/recordings/negative/{up['id']}/review", json={"review": True}).json()
    assert flagged["review"] is True
    assert client.put(f"/api/recordings/negative/{up['id']}/review", json={"review": False}).json()["review"] is False
    client.delete(f"/api/recordings/negative/{up['id']}")


def test_monitor_adopt_all_moves_files():
    from app import config
    from app.monitor import monitor_dir

    import numpy as np
    import soundfile as sf

    project = config.current_project()
    path = monitor_dir(project) / "20260101_000000_abcdef.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")
    res = client.post("/api/monitor/adopt-all").json()
    assert res["moved"] == 1
    negatives = client.get("/api/recordings?kind=negative").json()["items"]
    assert any(n["contributor"] == "monitor" for n in negatives)
    for n in negatives:
        client.delete(f"/api/recordings/negative/{n['id']}")
