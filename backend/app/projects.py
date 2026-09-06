"""Project routes: list / create / select / delete, ZIP export & import, sharing links."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .config import DATASETS_DIR, FEATURE_CACHE_DIR, current_project, get_project, list_projects, load_settings
from .jobs import _read_json, manager

router = APIRouter(prefix="/api", tags=["projects"])


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
        raise HTTPException(404, {"code": "not_found", "message": "Project not found"}) from None
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
        raise HTTPException(404, {"code": "not_found", "message": "Project not found"}) from None
    manager.load_project_state(current_project())
    return {"items": list_projects(), "current": current_project().id}


EXPORT_JOB_FILES = ("job.json", "result.json", "train.log", "training_parameters.yaml")


@router.get("/projects/{pid}/export")
def export_project(pid: str):
    """ZIP with recordings, settings and the light-weight outputs of every run (no feature caches / checkpoints)."""
    try:
        project = get_project(pid)
    except KeyError:
        raise HTTPException(404, {"code": "not_found", "message": "Project not found"}) from None
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
    for key in ("wake_word", "max_record_seconds", "contributor_target", "webhook_url"):
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
        raise HTTPException(404, {"code": "not_found", "message": "Project not found"}) from None
    token = set_share_token(project, bool(body.get("enabled", True)))
    return {"share_token": token}
