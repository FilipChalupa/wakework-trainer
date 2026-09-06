from app.export import esphome_snippet
from app.jobs import parse_minibatch_line, parse_validation_line


def test_parse_minibatch_line():
    line = "Validation Batch #3: Accuracy = 0.984; Recall = 0.900; Precision = 0.950; Loss = 0.0123; Mini-Batch #17"
    step, metrics = parse_minibatch_line(line, 250)
    assert step == 2 * 250 + 17
    assert metrics == {"accuracy": 0.984, "recall": 0.9, "precision": 0.95, "loss": 0.0123}
    assert parse_minibatch_line("something else", 250) is None


def test_parse_validation_line():
    line = (
        "INFO:absl:Step 250 (nonstreaming): Validation: recall at no faph = 72.222 with cutoff 0.05, accuracy = 97.99%, "
        "recall = 0.00%, precision = 0.00%, ambient false positives = 0, estimated false positives per hour = 0.00000, "
        "loss = 0.04130, auc = 0.99988, average viable recall = 0.740736127"
    )
    entry = parse_validation_line(line)
    assert entry["step"] == 250
    assert abs(entry["recall_at_no_faph"] - 0.72222) < 1e-6
    assert entry["accuracy"] == 0.9799
    assert entry["auc"] == 0.99988


def test_esphome_snippet_mentions_files():
    snippet = esphome_snippet("hey_jarvis", "hey jarvis")
    assert "hey_jarvis.json" in snippet and "micro_wake_word" in snippet


def test_interrupted_job_detected_after_restart(tmp_path):
    """A job without result.json and without a running manager is reported as interrupted/resumable."""
    import json

    from app.jobs import job_summary

    job_dir = tmp_path / "20260101_000000_abc"
    (job_dir / "features").mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({"job_id": "20260101_000000_abc", "slug": "x", "training": {"training_steps": 10}}))
    summary = job_summary(job_dir, running_job_id=None)
    assert summary["status"] == "interrupted" and summary["resumable"] is True
    assert job_summary(job_dir, running_job_id="20260101_000000_abc")["status"] == "running"


def test_webhook_payload(tmp_path, monkeypatch):
    """The webhook receives a JSON summary; network errors are swallowed (logged only)."""
    import json

    from app import jobs as training

    job_dir = tmp_path / "20260101_000000_abc"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({"job_id": "20260101_000000_abc", "wake_word": "hej", "slug": "hej"}))
    calls = []
    monkeypatch.setattr(training.requests, "post", lambda url, json, timeout: calls.append((url, json)) or type("R", (), {"status_code": 200})())
    monkeypatch.setenv("WEBHOOK_URL", "http://hook.invalid/x")
    monkeypatch.setenv("PUBLIC_URL", "https://trainer.example/")
    training.manager._notify_webhook(job_dir, "done", None)
    assert calls and calls[0][0] == "http://hook.invalid/x"
    payload = calls[0][1]
    assert payload["event"] == "training_finished" and payload["status"] == "done"
    assert payload["model_url"] == "https://trainer.example/api/jobs/20260101_000000_abc/model"

    def boom(*a, **k):
        raise OSError("unreachable")

    monkeypatch.setattr(training.requests, "post", boom)
    training.manager._notify_webhook(job_dir, "failed", "x")  # must not raise
