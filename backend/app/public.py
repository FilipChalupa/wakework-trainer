"""Token-protected public endpoints (no Basic auth): latest model/manifest for ESPHome `model: <url>` and
wake-word events posted by ESPHome devices (`on_wake_word_detected` → http_request)."""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .config import Project, current_project, find_project_by_model_token, get_or_create_model_token, load_settings

router = APIRouter(prefix="/api", tags=["public"])

MIN_ESPHOME = "2024.7.0"
_device_events: dict[str, deque[dict[str, Any]]] = {}
_lock = threading.Lock()


def _project(token: str) -> Project:
    project = find_project_by_model_token(token)
    if project is None:
        raise HTTPException(403, {"code": "bad_token", "message": "Unknown model token"})
    return project


def latest_done_job(project: Project) -> dict[str, Any] | None:
    from .training import list_jobs

    for job in list_jobs(project):
        if job["status"] == "done" and job["model_url"]:
            return job
    return None


def public_base(request: Request) -> str:
    import os

    base = os.environ.get("PUBLIC_URL", "").rstrip("/")
    if base:
        return base
    return str(request.base_url).rstrip("/")


def public_urls(request: Request, project: Project) -> dict[str, str]:
    token = get_or_create_model_token(project)
    base = public_base(request)
    return {
        "token": token,
        "manifest_url": f"{base}/api/public/{token}/manifest.json",
        "model_url": f"{base}/api/public/{token}/model.tflite",
        "device_event_url": f"{base}/api/public/{token}/device-event",
    }


def esphome_url_snippet(urls: dict[str, str], wake_word: str, slug: str) -> str:
    return f"""# ESPHome: load the latest "{wake_word}" model straight from Wake Word Trainer (re-flash after each retraining)
http_request:
  verify_ssl: false

micro_wake_word:
  models:
    - model: {urls['manifest_url']}
  on_wake_word_detected:
    - logger.log:
        format: "Wake word detected: %s"
        args: ['x.c_str()']
    # optional: report every detection back to the trainer's device test timeline
    - http_request.post:
        url: {urls['device_event_url']}
        request_headers:
          Content-Type: application/json
        body: !lambda |-
          return "{{\\"device\\": \\"" + App.get_name() + "\\", \\"wake_word\\": \\"" + x + "\\", \\"esphome_version\\": \\"" + std::string(ESPHOME_VERSION) + "\\"}}";
"""


def version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in str(text).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:3])


@router.get("/projects/{pid}/public-urls")
def get_public_urls(pid: str, request: Request):
    from .config import get_project

    try:
        project = get_project(pid)
    except KeyError:
        raise HTTPException(404, {"code": "not_found", "message": "Project not found"}) from None
    settings = load_settings(project)
    urls = public_urls(request, project)
    job = latest_done_job(project)
    target = job["target"] if job else settings["training"].get("target", "esphome")
    slug = job["slug"] if job else "wakeword"
    if target == "wyoming":
        from .training import wyoming_readme

        snippet = wyoming_readme(slug, settings["wake_word"]) + f"\nDirect download of the latest model: {urls['model_url']}\n"
    else:
        snippet = esphome_url_snippet(urls, settings["wake_word"], slug)
    return {
        **urls,
        "has_model": job is not None,
        "job_id": job["job_id"] if job else None,
        "target": target,
        "slug": slug,
        "minimum_esphome_version": MIN_ESPHOME,
        "snippet": snippet,
    }


@router.get("/public/{token}/manifest.json")
def public_manifest(token: str, request: Request):
    project = _project(token)
    job = latest_done_job(project)
    if not job:
        raise HTTPException(404, {"code": "no_models", "message": "No trained model yet"})
    job_dir = project.jobs_dir / job["job_id"]
    manifest_path = job_dir / f"{job['slug']}.json"
    if job.get("target") == "wyoming" or not manifest_path.exists():
        raise HTTPException(404, {"code": "no_manifest", "message": "The latest model is an openWakeWord (Wyoming) model – it has no ESPHome manifest; download model.tflite instead"})
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = f"{public_base(request)}/api/public/{token}/model.tflite"
    return JSONResponse(manifest, headers={"Cache-Control": "no-cache"})


@router.get("/public/{token}/model.tflite")
def public_model(token: str):
    project = _project(token)
    job = latest_done_job(project)
    if not job:
        raise HTTPException(404, {"code": "no_models", "message": "No trained model yet"})
    path = project.jobs_dir / job["job_id"] / f"{job['slug']}.tflite"
    return FileResponse(path, media_type="application/octet-stream", filename=path.name, headers={"Cache-Control": "no-cache"})


@router.post("/public/{token}/device-event")
async def device_event(token: str, request: Request):
    project = _project(token)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "device": str(body.get("device") or request.client.host if request.client else "device")[:64],
        "wake_word": str(body.get("wake_word") or "")[:64],
        "esphome_version": str(body.get("esphome_version") or "")[:32],
        "probability": body.get("probability"),
    }
    with _lock:
        _device_events.setdefault(project.id, deque(maxlen=500)).append(event)
    return PlainTextResponse("ok")


@router.get("/monitor/device-events")
def get_device_events(since: str | None = None):
    project = current_project()
    with _lock:
        events = list(_device_events.get(project.id, []))
    if since:
        events = [e for e in events if e["at"] > since]
    outdated = sorted({e["esphome_version"] for e in events if e["esphome_version"] and version_tuple(e["esphome_version"]) < version_tuple(MIN_ESPHOME)})
    return {"items": events[-200:], "minimum_esphome_version": MIN_ESPHOME, "outdated_versions": outdated, "now": datetime.now(timezone.utc).isoformat()}


@router.get("/bundle")
def bundle(request: Request, projects: str = ""):
    """ZIP with the latest model + manifest of several projects and one ESPHome YAML listing all of them."""
    import io
    import zipfile

    from fastapi.responses import StreamingResponse

    from .config import get_project, list_projects

    ids = [p for p in projects.split(",") if p] or [p["id"] for p in list_projects()]
    buffer = io.BytesIO()
    models_yaml = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for pid in ids:
            try:
                project = get_project(pid)
            except KeyError:
                continue
            job = latest_done_job(project)
            if not job:
                continue
            job_dir = project.jobs_dir / job["job_id"]
            slug = job["slug"]
            zf.write(job_dir / f"{slug}.tflite", f"{slug}.tflite")
            if (job_dir / f"{slug}.json").exists():
                zf.write(job_dir / f"{slug}.json", f"{slug}.json")
            models_yaml.append(f"    - model: {slug}.json   # {job['wake_word']}")
        if not models_yaml:
            raise HTTPException(404, {"code": "no_models", "message": "No trained models to bundle"})
        yaml_text = "# Several wake words on one ESPHome device – copy the .tflite/.json files next to this YAML.\nmicro_wake_word:\n  models:\n" + "\n".join(models_yaml) + "\n  on_wake_word_detected:\n    - logger.log:\n        format: \"Wake word detected: %s\"\n        args: ['x.c_str()']\n"
        zf.writestr("esphome-wake-words.yaml", yaml_text)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="wake-words-bundle.zip"'})

