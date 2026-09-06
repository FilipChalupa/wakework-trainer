"""Training job manager: runs the trainer as a subprocess, parses its output, streams it via SSE."""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from . import datasets
from .config import (
    DATASETS_DIR,
    FEATURE_CACHE_DIR,
    KEEP_JOBS,
    PROJECTS_DIR,
    Project,
    current_project,
    get_project,
    list_projects,
    load_settings,
    slugify,
)
from .recordings import list_recordings

router = APIRouter(prefix="/api", tags=["training"])

BACKEND_ROOT = Path(__file__).resolve().parent.parent

RE_MINIBATCH = re.compile(
    r"Validation Batch #(\d+): Accuracy = ([\d.]+); Recall = ([\d.]+); Precision = ([\d.]+); Loss = ([\d.]+); Mini-Batch #(\d+)"
)
RE_TRAIN_STEP = re.compile(
    r"Step #(\d+): rate ([\d.e+-]+), accuracy ([\d.]+)%, recall ([\d.]+)%, precision ([\d.]+)%, cross entropy ([\d.e+-]+)"
)
RE_VALIDATION = re.compile(
    r"Step (\d+) \(nonstreaming\): Validation: recall at no faph = ([\d.]+) with cutoff ([\d.]+), "
    r"accuracy = ([\d.]+)%, recall = ([\d.]+)%, precision = ([\d.]+)%, ambient false positives = (\d+), "
    r"estimated false positives per hour = ([\d.]+), loss = ([\d.e+-]+), auc = ([\d.]+), average viable recall = ([\d.e+-]+)"
)
RE_BEST = re.compile(r"So far the best minimization quantity is ([\d.]+) with best maximization quantity of ([\d.]+)%")

MAX_LOG_LINES = 600
RUNNING = ("downloading", "preparing", "training", "converting")
FINISHED = ("done", "failed", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_validation_line(line: str) -> dict[str, Any] | None:
    m = RE_VALIDATION.search(line)
    if not m:
        return None
    g = m.groups()
    return {
        "step": int(g[0]),
        "recall_at_no_faph": float(g[1]) / 100.0,
        "cutoff_for_no_faph": float(g[2]),
        "accuracy": float(g[3]) / 100.0,
        "recall": float(g[4]) / 100.0,
        "precision": float(g[5]) / 100.0,
        "ambient_false_positives": int(g[6]),
        "false_positives_per_hour": float(g[7]),
        "loss": float(g[8]),
        "auc": float(g[9]),
        "average_viable_recall": float(g[10]),
    }


def parse_minibatch_line(line: str, eval_interval: int) -> tuple[int, dict[str, float]] | None:
    m = RE_MINIBATCH.search(line)
    if not m:
        return None
    batch, acc, rec, prec, loss, mini = m.groups()
    step = (int(batch) - 1) * max(1, eval_interval) + int(mini)
    return step, {"accuracy": float(acc), "recall": float(rec), "precision": float(prec), "loss": float(loss)}


# ----- job discovery ----------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def find_job_dir(job_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_\-]+$", job_id or ""):
        raise HTTPException(400, "Bad job id")
    for entry in list_projects():
        candidate = PROJECTS_DIR / entry["id"] / "jobs" / job_id
        if candidate.is_dir():
            return candidate
    raise HTTPException(404, "Job not found")


def job_summary(job_dir: Path, running_job_id: str | None) -> dict[str, Any] | None:
    job = _read_json(job_dir / "job.json")
    if not job:
        return None
    result = _read_json(job_dir / "result.json")
    slug = job.get("slug", "wakeword")
    model = job_dir / f"{slug}.tflite"
    status = result.get("status")
    if status is None:
        status = "running" if running_job_id == job["job_id"] else "interrupted"
    return {
        "job_id": job["job_id"],
        "project_id": job.get("project_id"),
        "wake_word": job.get("wake_word"),
        "label": job.get("label") or "",
        "overrides": job.get("overrides") or {},
        "slug": slug,
        "created_at": job.get("created_at"),
        "finished_at": result.get("finished_at"),
        "status": status,
        "positive_count": job.get("positive_count"),
        "training": job.get("training"),
        "final_metrics": result.get("final_metrics"),
        "validation_last": (result.get("validation") or [None])[-1],
        "model_url": f"/api/jobs/{job['job_id']}/model" if model.exists() else None,
        "manifest_url": f"/api/jobs/{job['job_id']}/manifest" if (job_dir / f"{slug}.json").exists() else None,
        "export_url": f"/api/jobs/{job['job_id']}/export" if model.exists() else None,
        "model_size": model.stat().st_size if model.exists() else None,
        "resumable": status == "interrupted" and (job_dir / "features").is_dir(),
    }


def list_jobs(project: Project, running_job_id: str | None = None) -> list[dict[str, Any]]:
    project.ensure()
    running = running_job_id
    if running is None:
        mgr = globals().get("manager")
        if mgr is not None and mgr.is_running():
            running = mgr.state.get("job_id")
    jobs = []
    for job_dir in sorted(project.jobs_dir.iterdir(), reverse=True):
        if job_dir.is_dir():
            summary = job_summary(job_dir, running)
            if summary:
                jobs.append(summary)
    return jobs


def prune_jobs(project: Project, keep: int = KEEP_JOBS) -> int:
    """Deletes the oldest finished jobs beyond ``keep``. Returns the number removed."""
    finished = [j for j in list_jobs(project) if j["status"] in FINISHED]
    removed = 0
    for job in finished[keep:]:
        shutil.rmtree(project.jobs_dir / job["job_id"], ignore_errors=True)
        removed += 1
    return removed


# ----- manager ------------------------------------------------------------------------
class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.state: dict[str, Any] = self._idle_state()
        self.log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._last_minibatch_emit = 0.0
        self.queue: list[dict[str, Any]] = []
        self._restore_last_job()

    @staticmethod
    def _idle_state() -> dict[str, Any]:
        return {
            "status": "idle",  # idle | downloading | preparing | training | converting | done | failed | cancelled | interrupted
            "job_id": None,
            "project_id": None,
            "label": "",
            "wake_word": None,
            "stage": None,
            "stage_key": None,
            "message": None,
            "message_key": None,
            "message_params": None,
            "progress": {"current": 0, "total": 0},
            "step": 0,
            "total_steps": 0,
            "eval_step_interval": 0,
            "train_metrics": None,
            "validation": [],
            "best": None,
            "final_metrics": None,
            "model_url": None,
            "manifest_url": None,
            "export_url": None,
            "resumable": False,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def _restore_last_job(self) -> None:
        """After a restart, show the outcome of the most recent job of the current project."""
        try:
            self.load_project_state(current_project())
        except Exception:  # noqa: BLE001  (best effort only)
            pass

    def load_project_state(self, project: Project) -> None:
        """Replaces the idle/finished state with the last job of ``project`` (no-op while running)."""
        if self.is_running():
            return
        with self._lock:
            self.state = self._idle_state()
            self.state["project_id"] = project.id
        self.log.clear()
        jobs = list_jobs(project, running_job_id=self.state.get("job_id") if self.is_running() else None)
        if not jobs:
            self._publish("snapshot", self.snapshot())
            return
        job = jobs[0]
        job_dir = project.jobs_dir / job["job_id"]
        result = _read_json(job_dir / "result.json")
        total = int((job.get("training") or {}).get("training_steps", 0))
        status = job["status"]
        stage = {"done": "Done", "failed": "Error", "cancelled": "Cancelled", "interrupted": "Interrupted"}.get(status, status)
        with self._lock:
            self.state.update({
                "status": status,
                "job_id": job["job_id"],
                "wake_word": job.get("wake_word"),
                "stage": stage,
                "stage_key": status,
                "progress": {"current": total if status == "done" else 0, "total": total},
                "step": total if status == "done" else 0,
                "total_steps": total,
                "validation": result.get("validation") or [],
                "best": result.get("best"),
                "final_metrics": result.get("final_metrics"),
                "error": result.get("error"),
                "model_url": job["model_url"],
                "manifest_url": job["manifest_url"],
                "export_url": job["export_url"],
                "resumable": job["resumable"],
                "started_at": job.get("created_at"),
                "finished_at": result.get("finished_at"),
            })
        log_file = job_dir / "train.log"
        if log_file.is_file():
            self.log.extend(log_file.read_text().splitlines()[-MAX_LOG_LINES:])
        self._publish("snapshot", self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snap = json.loads(json.dumps(self.state))
            snap["queue"] = list(self.queue)
        snap["log_tail"] = list(self.log)[-80:]
        return snap

    def is_running(self) -> bool:
        return self.state["status"] in RUNNING

    # ----- pub/sub ---------------------------------------------------------
    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append((loop, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(l, q) for l, q in self._subscribers if q is not queue]

    def _publish(self, event: str, data: dict[str, Any]) -> None:
        payload = {"event": event, "data": data}
        with self._lock:
            subs = list(self._subscribers)
        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(self._put_nowait, queue, payload)
            except RuntimeError:
                pass

    @staticmethod
    def _put_nowait(queue: asyncio.Queue, payload: dict) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def _update(self, **fields: Any) -> None:
        with self._lock:
            self.state.update(fields)
        self._publish("state", fields)

    def _log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        self.log.append(line)
        self._publish("log", {"line": line})

    # ----- job lifecycle -------------------------------------------------
    def start(self, overrides: dict[str, Any] | None = None, label: str | None = None) -> dict[str, Any]:
        """Starts a run now, or queues it when one is already running."""
        project = current_project()
        positives = list_recordings("positive", project)
        if len(positives) < 3:
            raise HTTPException(400, {"code": "too_few_samples", "message": "Record at least 3 wake word samples (20-40 recommended)."})
        spec = {"project_id": project.id, "overrides": overrides or {}, "label": (label or "").strip()[:60], "queued_at": _now(), "id": uuid.uuid4().hex[:8]}
        if self.is_running():
            with self._lock:
                self.queue.append(spec)
            self._publish("queue", {"queue": self.queue})
            return self.snapshot()
        self._start_spec(spec)
        return self.snapshot()

    def _start_spec(self, spec: dict[str, Any]) -> None:
        project = Project(spec["project_id"]).ensure()
        positives = list_recordings("positive", project)
        settings = load_settings(project)
        training = dict(settings["training"])
        for key, value in (spec.get("overrides") or {}).items():
            if key in training and value is not None:
                training[key] = type(training[key])(value)
        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_dir = project.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": job_id,
            "project_id": project.id,
            "wake_word": settings["wake_word"],
            "slug": slugify(settings["wake_word"]),
            "training": training,
            "overrides": spec.get("overrides") or {},
            "label": spec.get("label") or "",
            "positive_dir": str(project.positive_dir),
            "negative_dir": str(project.negative_dir),
            "datasets_dir": str(DATASETS_DIR),
            "feature_cache_dir": str(FEATURE_CACHE_DIR),
            "job_dir": str(job_dir),
            "positive_count": len(positives),
            "negative_count": len(list_recordings("negative", project)),
            "created_at": _now(),
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False))
        removed = prune_jobs(project)
        self._launch(job, resume=False)
        if removed:
            self._log(f"Removed {removed} old training run(s) (KEEP_JOBS={KEEP_JOBS})")

    def sweep(self, param: str, values: list[Any], label: str | None = None) -> dict[str, Any]:
        from .config import DEFAULT_TRAINING

        if param not in DEFAULT_TRAINING or param == "hard_negatives":
            raise HTTPException(400, {"code": "bad_param", "message": f"Cannot sweep '{param}'"})
        values = [v for v in values if v is not None][:8]
        if not values:
            raise HTTPException(400, {"code": "bad_values", "message": "No values given"})
        for value in values:
            self.start({param: value}, f"{label or 'sweep'} {param}={value}")
        return self.snapshot()

    def drop_queued(self, spec_id: str) -> None:
        with self._lock:
            self.queue = [q for q in self.queue if q["id"] != spec_id]
        self._publish("queue", {"queue": self.queue})

    def _start_next(self) -> None:
        with self._lock:
            spec = self.queue.pop(0) if self.queue else None
        self._publish("queue", {"queue": self.queue})
        if spec is None:
            return
        try:
            self._start_spec(spec)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Could not start queued run: {exc}")
            self._start_next()

    def resume(self) -> dict[str, Any]:
        if self.is_running():
            raise HTTPException(409, {"code": "already_running", "message": "Training is already running"})
        job_id = self.state.get("job_id")
        if self.state.get("status") != "interrupted" or not job_id:
            raise HTTPException(409, {"code": "nothing_to_resume", "message": "No interrupted training run to resume"})
        job = _read_json(find_job_dir(job_id) / "job.json")
        if not job:
            raise HTTPException(404, "Job not found")
        self._launch(job, resume=True)
        return self.snapshot()

    def _launch(self, job: dict[str, Any], resume: bool) -> None:
        self.log.clear()
        self._cancel.clear()
        with self._lock:
            self.state = self._idle_state()
        self._update(
            status="downloading",
            job_id=job["job_id"],
            project_id=job.get("project_id"),
            label=job.get("label") or "",
            wake_word=job["wake_word"],
            stage="Checking datasets",
            stage_key="checking_datasets",
            started_at=_now(),
            total_steps=int(job["training"]["training_steps"]),
            eval_step_interval=int(job["training"]["eval_step_interval"]),
        )
        self._thread = threading.Thread(target=self._run, args=(job, resume), daemon=True)
        self._thread.start()

    def cancel(self) -> dict[str, Any]:
        if not self.is_running():
            raise HTTPException(409, {"code": "not_running", "message": "No training running"})
        self._cancel.set()
        with self._lock:
            self.queue.clear()
        self._publish("queue", {"queue": []})
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
        return self.snapshot()

    def _run(self, job: dict[str, Any], resume: bool) -> None:
        job_dir = Path(job["job_dir"])
        try:
            self._ensure_datasets()
            if self._cancel.is_set():
                raise InterruptedError
            self._update(status="preparing", stage="Preparing data", stage_key="preparing", progress={"current": 0, "total": 0})
            env = dict(os.environ)
            env.update({"PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONPATH": str(BACKEND_ROOT)})
            cmd = [sys.executable, "-m", "trainer.run", "--job", str(job_dir / "job.json")]
            if resume:
                cmd.append("--resume")
            self._log(f"$ {' '.join(cmd)}")
            self._proc = subprocess.Popen(cmd, cwd=str(job_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            self._pump(self._proc)
            code = self._proc.wait()
            if self._cancel.is_set():
                raise InterruptedError
            if code != 0:
                raise RuntimeError(f"Trainer exited with code {code}")
            self._finish(job)
        except InterruptedError:
            self._update(status="cancelled", stage="Cancelled", stage_key="cancelled", finished_at=_now())
            self._write_result(job_dir, "cancelled")
        except Exception as exc:  # noqa: BLE001
            self._log(f"ERROR: {exc}")
            self._update(status="failed", stage="Error", stage_key="failed", error=str(exc), finished_at=_now())
            self._write_result(job_dir, "failed", error=str(exc))
        finally:
            self._proc = None
            if not self._cancel.is_set():
                self._start_next()

    def _ensure_datasets(self) -> None:
        for name, meta in datasets.DATASETS.items():
            if not meta["required"] or datasets.is_installed(name):
                continue
            self._update(
                status="downloading",
                stage=f"Downloading {meta['title']}",
                stage_key="downloading_dataset",
                message=None,
                message_key=None,
                message_params={"title": meta["title"]},
                progress={"current": 0, "total": meta["size_mb"] << 20},
            )
            self._log(f"Downloading dataset {name} from {meta['url']}")

            def progress(info: dict) -> None:
                if "received" in info:
                    self._update(progress={"current": info["received"], "total": info.get("total") or (meta["size_mb"] << 20)})
                if info.get("state") == "extracting":
                    self._update(stage=f"Extracting {meta['title']}", stage_key="extracting_dataset", message_params={"title": meta["title"]})

            datasets.download(name, progress)
            self._log(f"Dataset {name} ready")

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                idx_n = buf.find(b"\n")
                idx_r = buf.find(b"\r")
                candidates = [i for i in (idx_n, idx_r) if i >= 0]
                if not candidates:
                    break
                idx = min(candidates)
                line = buf[:idx].decode("utf-8", errors="replace")
                buf = buf[idx + 1:]
                self._handle_line(line)
        if buf.strip():
            self._handle_line(buf.decode("utf-8", errors="replace"))

    def _handle_line(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        if line.startswith("@@"):
            try:
                ev = json.loads(line[2:])
            except json.JSONDecodeError:
                self._log(line)
                return
            self._handle_event(ev)
            return
        parsed = parse_minibatch_line(line, int(self.state["eval_step_interval"]) or 1)
        if parsed:
            step, metrics = parsed
            now = time.time()
            if now - self._last_minibatch_emit > 0.25 or step % (int(self.state["eval_step_interval"]) or 1) == 0:
                self._last_minibatch_emit = now
                self._update(step=step, train_metrics=metrics, progress={"current": step, "total": int(self.state["total_steps"])})
            return
        entry = parse_validation_line(line)
        if entry:
            with self._lock:
                self.state["validation"].append(entry)
            self._publish("state", {"validation": self.state["validation"]})
            self._log(line)
            return
        m = RE_BEST.search(line)
        if m:
            self._update(best={"minimization": float(m.group(1)), "maximization": float(m.group(2)) / 100.0})
        self._log(line)

    def _handle_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        if kind == "stage":
            fields: dict[str, Any] = {
                "stage": ev.get("name"),
                "stage_key": ev.get("key"),
                "message": ev.get("message"),
                "message_key": ev.get("message_key"),
                "message_params": ev.get("params"),
            }
            if ev.get("status"):
                fields["status"] = ev["status"]
            if "total" in ev:
                fields["progress"] = {"current": ev.get("current", 0), "total": ev.get("total", 0)}
            self._update(**fields)
            if ev.get("message"):
                self._log(f"[{ev.get('name')}] {ev['message']}")
        elif kind == "progress":
            self._update(progress={"current": ev.get("current", 0), "total": ev.get("total", 0)})
        elif kind == "log":
            self._log(str(ev.get("message", "")))
        elif kind == "training_config":
            self._update(total_steps=int(ev.get("total_steps", 0)), eval_step_interval=int(ev.get("eval_step_interval", 1)))
        elif kind == "final_metrics":
            self._update(final_metrics=ev.get("summary"))
        elif kind == "done":
            self._update(stage="Done", stage_key="done", message=ev.get("message"), message_key=ev.get("message_key"), message_params=ev.get("params"))

    def _finish(self, job: dict[str, Any]) -> None:
        job_dir = Path(job["job_dir"])
        model = job_dir / f"{job['slug']}.tflite"
        manifest = job_dir / f"{job['slug']}.json"
        if not model.exists():
            raise RuntimeError("Trainer finished but no .tflite model was produced")
        self._update(
            status="done",
            stage="Done",
            stage_key="done",
            progress={"current": self.state["total_steps"], "total": self.state["total_steps"]},
            model_url=f"/api/jobs/{job['job_id']}/model",
            manifest_url=f"/api/jobs/{job['job_id']}/manifest" if manifest.exists() else None,
            export_url=f"/api/jobs/{job['job_id']}/export",
            finished_at=_now(),
        )
        self._write_result(job_dir, "done")

    def _write_result(self, job_dir: Path, status: str, error: str | None = None) -> None:
        result = {
            "status": status,
            "error": error,
            "finished_at": _now(),
            "validation": self.state.get("validation"),
            "final_metrics": self.state.get("final_metrics"),
            "best": self.state.get("best"),
        }
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (job_dir / "train.log").write_text("\n".join(self.log))
        threading.Thread(target=self._notify_webhook, args=(job_dir, status, error), daemon=True).start()

    def _notify_webhook(self, job_dir: Path, status: str, error: str | None) -> None:
        """POSTs a JSON summary to the project's webhook (or WEBHOOK_URL env) – e.g. a Home Assistant webhook."""
        job = _read_json(job_dir / "job.json")
        url = ""
        try:
            if job.get("project_id"):
                url = load_settings(Project(job["project_id"])).get("webhook_url") or ""
        except Exception:  # noqa: BLE001
            url = ""
        url = url or os.environ.get("WEBHOOK_URL", "").strip()
        if not url:
            return
        base = os.environ.get("PUBLIC_URL", "").rstrip("/")
        payload = {
            "event": "training_finished",
            "status": status,
            "error": error,
            "project_id": job.get("project_id"),
            "wake_word": job.get("wake_word"),
            "job_id": job.get("job_id"),
            "model_url": f"{base}/api/jobs/{job.get('job_id')}/model" if status == "done" else None,
            "final_metrics": self.state.get("final_metrics"),
            "finished_at": _now(),
        }
        try:
            requests.post(url, json=payload, timeout=10)
            self._log(f"Webhook notified: {url}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Webhook failed: {exc}")


manager = JobManager()


# ----- routes ---------------------------------------------------------------
@router.post("/train")
async def start_training(request: Request):
    body: dict[str, Any] = {}
    try:
        if int(request.headers.get("content-length") or 0) > 0:
            body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    return manager.start(body.get("training") or {}, body.get("label"))


@router.post("/train/sweep")
async def start_sweep(body: dict[str, Any]):
    return manager.sweep(str(body.get("param", "")), list(body.get("values") or []), body.get("label"))


@router.get("/train/queue")
def get_queue():
    return {"queue": manager.queue}


@router.delete("/train/queue/{spec_id}")
def delete_queued(spec_id: str):
    manager.drop_queued(spec_id)
    return {"queue": manager.queue}


@router.post("/train/resume")
def resume_training():
    return manager.resume()


@router.post("/train/cancel")
def cancel_training():
    return manager.cancel()


@router.get("/train")
def training_snapshot():
    return manager.snapshot()


@router.get("/train/status")
async def training_status_stream():
    loop = asyncio.get_running_loop()
    queue = manager.subscribe(loop)

    async def gen():
        try:
            yield _sse("snapshot", manager.snapshot())
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(item["event"], item["data"])
        finally:
            manager.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/train/model")
def latest_model():
    for job in list_jobs(current_project()):
        if job["status"] == "done" and job["model_url"]:
            return job_model(job["job_id"])
    raise HTTPException(404, "No trained model yet")


@router.get("/jobs")
def get_jobs():
    return {"items": list_jobs(current_project())}


def _job_file(job_id: str, suffix: str) -> tuple[Path, dict[str, Any]]:
    job_dir = find_job_dir(job_id)
    job = _read_json(job_dir / "job.json")
    path = job_dir / f"{job.get('slug', 'wakeword')}{suffix}"
    if not path.exists():
        raise HTTPException(404, "File not found")
    return path, job


@router.get("/jobs/{job_id}/model")
def job_model(job_id: str):
    path, _ = _job_file(job_id, ".tflite")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/jobs/{job_id}/manifest")
def job_manifest(job_id: str):
    path, _ = _job_file(job_id, ".json")
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str):
    log = find_job_dir(job_id) / "train.log"
    if not log.exists():
        raise HTTPException(404, "Log not found")
    return FileResponse(log, media_type="text/plain")


def esphome_snippet(slug: str, wake_word: str) -> str:
    return f"""# Example ESPHome configuration for the "{wake_word}" wake word.
# Copy {slug}.tflite and {slug}.json next to this YAML (or point `model:` to a URL).
micro_wake_word:
  models:
    - model: {slug}.json
  on_wake_word_detected:
    - logger.log:
        format: "Wake word detected: %s"
        args: ['x.c_str()']
    # - voice_assistant.start:
    #     wake_word: !lambda return x;

# Tuning: if "{wake_word}" is hard to trigger, lower "probability_cutoff" in {slug}.json;
# if it triggers falsely, raise it (0.5 - 0.99).
"""


@router.get("/jobs/{job_id}/export")
def job_export(job_id: str):
    model, job = _job_file(job_id, ".tflite")
    job_dir = model.parent
    slug = job.get("slug", "wakeword")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model, f"{slug}/{model.name}")
        manifest = job_dir / f"{slug}.json"
        if manifest.exists():
            zf.write(manifest, f"{slug}/{manifest.name}")
        zf.writestr(f"{slug}/esphome-example.yaml", esphome_snippet(slug, job.get("wake_word", slug)))
        for extra in ("training_parameters.yaml", "result.json", "train.log"):
            if (job_dir / extra).exists():
                zf.write(job_dir / extra, f"{slug}/training/{extra}")
        readme = (
            f"Wake word model '{job.get('wake_word')}' trained with Wake Word Trainer / microWakeWord on {job.get('created_at')}.\n\n"
            f"Files:\n  {slug}.tflite         quantised streaming TensorFlow Lite model\n  {slug}.json           ESPHome micro_wake_word manifest\n"
            f"  esphome-example.yaml  example ESPHome configuration\n  training/             training parameters, metrics and log\n"
        )
        zf.writestr(f"{slug}/README.txt", readme)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{slug}-wakeword.zip"'})


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job_dir = find_job_dir(job_id)
    if manager.is_running() and manager.state.get("job_id") == job_id:
        raise HTTPException(409, {"code": "already_running", "message": "Job is running"})
    shutil.rmtree(job_dir)
    if manager.state.get("job_id") == job_id:
        manager.load_project_state(current_project())
    return {"deleted": job_id}


# ----- projects -----------------------------------------------------------------
@router.get("/projects")
def get_projects():
    return {"items": list_projects(), "current": current_project().id}


@router.post("/projects")
async def post_project(body: dict[str, Any]):
    from .config import create_project, select_project

    name = str(body.get("name", "")).strip()
    if not name:
        raise HTTPException(400, {"code": "name_required", "message": "Project name is required"})
    if manager.is_running():
        raise HTTPException(409, {"code": "already_running", "message": "Cannot switch projects while training"})
    project = create_project(name, str(body.get("wake_word") or name))
    select_project(project.id)
    manager.load_project_state(project)
    return {"items": list_projects(), "current": project.id}


@router.post("/projects/{pid}/select")
def post_select(pid: str):
    from .config import select_project

    if manager.is_running():
        raise HTTPException(409, {"code": "already_running", "message": "Cannot switch projects while training"})
    try:
        project = select_project(pid)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    manager.load_project_state(project)
    return {"items": list_projects(), "current": project.id}


@router.delete("/projects/{pid}")
def delete_project_route(pid: str):
    from .config import delete_project

    if manager.is_running():
        raise HTTPException(409, {"code": "already_running", "message": "Cannot delete projects while training"})
    try:
        get_project(pid)
        delete_project(pid)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    manager.load_project_state(current_project())
    return {"items": list_projects(), "current": current_project().id}


EXPORT_JOB_FILES = ("job.json", "result.json", "train.log", "training_parameters.yaml")


@router.get("/projects/{pid}/export")
def export_project(pid: str):
    """ZIP with recordings, settings and the light-weight outputs of every run (no feature caches / checkpoints)."""
    try:
        project = get_project(pid)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    settings = load_settings(project)
    settings["share_token"] = None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(settings, indent=2, ensure_ascii=False))
        for kind in ("positive_samples", "negative_samples"):
            folder = project.dir / kind
            for f in sorted(folder.glob("*.wav")) + [folder / "meta.json"]:
                if f.exists():
                    zf.write(f, f"{kind}/{f.name}")
        for job_dir in sorted(project.jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            job = _read_json(job_dir / "job.json")
            slug = job.get("slug", "wakeword")
            for name in EXPORT_JOB_FILES + (f"{slug}.tflite", f"{slug}.json"):
                if (job_dir / name).exists():
                    zf.write(job_dir / name, f"jobs/{job_dir.name}/{name}")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{pid}-project.zip"'})


@router.post("/projects/import")
async def import_project(file: UploadFile = File(...)):
    from .config import create_project, select_project, write_settings

    if manager.is_running():
        raise HTTPException(409, {"code": "already_running", "message": "Cannot import while training"})
    raw = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        settings = json.loads(zf.read("project.json"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(400, {"code": "bad_archive", "message": f"Not a project archive: {exc}"}) from exc
    project = create_project(str(settings.get("name") or settings.get("wake_word") or "imported"), str(settings.get("wake_word") or "wakeword"))
    root = project.dir.resolve()
    for member in zf.infolist():
        if member.is_dir():
            continue
        target = (root / member.filename).resolve()
        if root not in target.parents or member.filename == "project.json":
            continue
        top = member.filename.split("/")[0]
        if top not in ("positive_samples", "negative_samples", "jobs"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(member))
    # rewrite absolute paths inside job.json files to the new location
    for job_file in project.jobs_dir.glob("*/job.json"):
        job = _read_json(job_file)
        job.update({
            "project_id": project.id,
            "positive_dir": str(project.positive_dir),
            "negative_dir": str(project.negative_dir),
            "datasets_dir": str(DATASETS_DIR),
            "feature_cache_dir": str(FEATURE_CACHE_DIR),
            "job_dir": str(job_file.parent),
        })
        job_file.write_text(json.dumps(job, indent=2, ensure_ascii=False))
    merged = load_settings(project)
    for key in ("wake_word", "sample_duration_s", "contributor_target", "webhook_url"):
        if key in settings:
            merged[key] = settings[key]
    merged["training"].update(settings.get("training", {}))
    merged["share_token"] = None
    write_settings(project, merged)
    select_project(project.id)
    manager.load_project_state(project)
    return {"items": list_projects(), "current": project.id}


@router.post("/projects/{pid}/share")
async def post_share(pid: str, body: dict[str, Any]):
    from .config import set_share_token

    try:
        project = get_project(pid)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    token = set_share_token(project, bool(body.get("enabled", True)))
    return {"share_token": token}
