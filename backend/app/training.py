"""Training job manager: runs the trainer as a subprocess, parses its output, streams it via SSE."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from . import datasets
from .config import (
    DATASETS_DIR,
    FEATURE_CACHE_DIR,
    JOBS_DIR,
    NEGATIVE_DIR,
    POSITIVE_DIR,
    load_project,
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
RE_FINAL = re.compile(r"Final TFLite model on the testing set: (.*)")
RE_BEST = re.compile(r"So far the best minimization quantity is ([\d.]+) with best maximization quantity of ([\d.]+)%")

MAX_LOG_LINES = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # ----- state helpers -------------------------------------------------
    @staticmethod
    def _idle_state() -> dict[str, Any]:
        return {
            "status": "idle",  # idle | downloading | preparing | training | converting | done | failed | cancelled
            "job_id": None,
            "wake_word": None,
            "stage": None,
            "message": None,
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
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snap = json.loads(json.dumps(self.state))
        snap["log_tail"] = list(self.log)[-80:]
        return snap

    def is_running(self) -> bool:
        return self.state["status"] in ("downloading", "preparing", "training", "converting")

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
    def start(self) -> dict[str, Any]:
        if self.is_running():
            raise HTTPException(409, "Training is already running")
        positives = list_recordings("positive")
        if len(positives) < 3:
            raise HTTPException(400, "Nahrajte alespoň 3 vzorky wake wordu (doporučeno 20–40).")
        project = load_project()
        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(project["wake_word"])
        job = {
            "job_id": job_id,
            "wake_word": project["wake_word"],
            "slug": slug,
            "training": project["training"],
            "positive_dir": str(POSITIVE_DIR),
            "negative_dir": str(NEGATIVE_DIR),
            "datasets_dir": str(DATASETS_DIR),
            "feature_cache_dir": str(FEATURE_CACHE_DIR),
            "job_dir": str(job_dir),
            "positive_count": len(positives),
            "negative_count": len(list_recordings("negative")),
            "created_at": _now(),
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False))
        self.log.clear()
        self._cancel.clear()
        with self._lock:
            self.state = self._idle_state()
        self._update(
            status="downloading",
            job_id=job_id,
            wake_word=project["wake_word"],
            stage="Kontrola datasetů",
            message=None,
            started_at=_now(),
            total_steps=int(project["training"]["training_steps"]),
            eval_step_interval=int(project["training"]["eval_step_interval"]),
        )
        self._thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        self._thread.start()
        return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        if not self.is_running():
            raise HTTPException(409, "No training running")
        self._cancel.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
        return self.snapshot()

    def _run(self, job: dict[str, Any]) -> None:
        job_dir = Path(job["job_dir"])
        try:
            self._ensure_datasets()
            if self._cancel.is_set():
                raise InterruptedError
            self._update(status="preparing", stage="Příprava dat", progress={"current": 0, "total": 0})
            env = dict(os.environ)
            env.update({
                "PYTHONUNBUFFERED": "1",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONPATH": str(BACKEND_ROOT),
            })
            cmd = [sys.executable, "-m", "trainer.run", "--job", str(job_dir / "job.json")]
            self._log(f"$ {' '.join(cmd)}")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(job_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            self._pump(self._proc)
            code = self._proc.wait()
            if self._cancel.is_set():
                raise InterruptedError
            if code != 0:
                raise RuntimeError(f"Trainer exited with code {code}")
            self._finish(job)
        except InterruptedError:
            self._update(status="cancelled", stage="Zrušeno", finished_at=_now())
            self._write_result(job_dir, "cancelled")
        except Exception as exc:  # noqa: BLE001
            self._log(f"ERROR: {exc}")
            self._update(status="failed", stage="Chyba", error=str(exc), finished_at=_now())
            self._write_result(job_dir, "failed", error=str(exc))
        finally:
            self._proc = None

    def _ensure_datasets(self) -> None:
        for name, meta in datasets.DATASETS.items():
            if not meta["required"] or datasets.is_installed(name):
                continue
            self._update(status="downloading", stage=f"Stahuji {meta['title']}", progress={"current": 0, "total": meta["size_mb"] << 20})
            self._log(f"Downloading dataset {name} from {meta['url']}")

            def progress(info: dict) -> None:
                if "received" in info:
                    self._update(progress={"current": info["received"], "total": info.get("total") or (meta["size_mb"] << 20)})
                if info.get("state") == "extracting":
                    self._update(stage=f"Rozbaluji {meta['title']}")

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
        m = RE_MINIBATCH.search(line)
        if m:
            batch, acc, rec, prec, loss, mini = m.groups()
            interval = int(self.state["eval_step_interval"]) or 1
            step = (int(batch) - 1) * interval + int(mini)
            now = time.time()
            if now - self._last_minibatch_emit > 0.25 or int(mini) == 0:
                self._last_minibatch_emit = now
                self._update(
                    step=step,
                    train_metrics={"accuracy": float(acc), "recall": float(rec), "precision": float(prec), "loss": float(loss)},
                    progress={"current": step, "total": int(self.state["total_steps"])},
                )
            return
        m = RE_VALIDATION.search(line)
        if m:
            g = m.groups()
            entry = {
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
            with self._lock:
                self.state["validation"].append(entry)
            self._publish("state", {"validation": self.state["validation"]})
            self._log(line)
            return
        m = RE_BEST.search(line)
        if m:
            self._update(best={"minimization": float(m.group(1)), "maximization": float(m.group(2)) / 100.0})
            self._log(line)
            return
        m = RE_FINAL.search(line)
        if m:
            self._update(final_metrics=m.group(1).strip())
            self._log(line)
            return
        if RE_TRAIN_STEP.search(line):
            self._log(line)
            return
        self._log(line)

    def _handle_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        if kind == "stage":
            fields: dict[str, Any] = {"stage": ev.get("name"), "message": ev.get("message")}
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
            self._update(final_metrics=str(ev.get("summary", "")))
        elif kind == "done":
            self._update(stage="Hotovo", message=ev.get("message"))

    def _finish(self, job: dict[str, Any]) -> None:
        job_dir = Path(job["job_dir"])
        model = job_dir / f"{job['slug']}.tflite"
        manifest = job_dir / f"{job['slug']}.json"
        if not model.exists():
            raise RuntimeError("Trainer finished but no .tflite model was produced")
        self._update(
            status="done",
            stage="Hotovo",
            progress={"current": self.state["total_steps"], "total": self.state["total_steps"]},
            model_url=f"/api/jobs/{job['job_id']}/model",
            manifest_url=f"/api/jobs/{job['job_id']}/manifest" if manifest.exists() else None,
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


manager = JobManager()


# ----- jobs listing ---------------------------------------------------------
def list_jobs() -> list[dict[str, Any]]:
    jobs = []
    for job_dir in sorted(JOBS_DIR.iterdir(), reverse=True):
        job_file = job_dir / "job.json"
        if not job_file.is_file():
            continue
        try:
            job = json.loads(job_file.read_text())
        except json.JSONDecodeError:
            continue
        result = {}
        result_file = job_dir / "result.json"
        if result_file.exists():
            try:
                result = json.loads(result_file.read_text())
            except json.JSONDecodeError:
                result = {}
        model = job_dir / f"{job.get('slug', 'wakeword')}.tflite"
        status = result.get("status")
        if status is None:
            status = "running" if manager.state.get("job_id") == job["job_id"] and manager.is_running() else "failed"
        jobs.append({
            "job_id": job["job_id"],
            "wake_word": job.get("wake_word"),
            "slug": job.get("slug"),
            "created_at": job.get("created_at"),
            "finished_at": result.get("finished_at"),
            "status": status,
            "positive_count": job.get("positive_count"),
            "training": job.get("training"),
            "final_metrics": result.get("final_metrics"),
            "model_url": f"/api/jobs/{job['job_id']}/model" if model.exists() else None,
            "manifest_url": f"/api/jobs/{job['job_id']}/manifest" if (job_dir / f"{job.get('slug', 'wakeword')}.json").exists() else None,
            "model_size": model.stat().st_size if model.exists() else None,
        })
    return jobs


def _job_dir(job_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_\-]+$", job_id):
        raise HTTPException(400, "Bad job id")
    path = JOBS_DIR / job_id
    if not path.is_dir():
        raise HTTPException(404, "Job not found")
    return path


# ----- routes ---------------------------------------------------------------
@router.post("/train")
def start_training():
    return manager.start()


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

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/train/model")
def latest_model():
    for job in list_jobs():
        if job["status"] == "done" and job["model_url"]:
            return job_model(job["job_id"])
    raise HTTPException(404, "No trained model yet")


@router.get("/jobs")
def get_jobs():
    return {"items": list_jobs()}


@router.get("/jobs/{job_id}/model")
def job_model(job_id: str):
    job_dir = _job_dir(job_id)
    job = json.loads((job_dir / "job.json").read_text())
    model = job_dir / f"{job['slug']}.tflite"
    if not model.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(model, media_type="application/octet-stream", filename=model.name)


@router.get("/jobs/{job_id}/manifest")
def job_manifest(job_id: str):
    job_dir = _job_dir(job_id)
    job = json.loads((job_dir / "job.json").read_text())
    manifest = job_dir / f"{job['slug']}.json"
    if not manifest.exists():
        raise HTTPException(404, "Manifest not found")
    return FileResponse(manifest, media_type="application/json", filename=manifest.name)


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str):
    job_dir = _job_dir(job_id)
    log = job_dir / "train.log"
    if not log.exists():
        raise HTTPException(404, "Log not found")
    return FileResponse(log, media_type="text/plain")


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job_dir = _job_dir(job_id)
    if manager.is_running() and manager.state.get("job_id") == job_id:
        raise HTTPException(409, "Job is running")
    shutil.rmtree(job_dir)
    return {"deleted": job_id}
