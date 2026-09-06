"""Record-only access for contributors via a shared link (token), no login required."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from . import recordings
from .config import Project, find_project_by_token, load_settings, slugify

router = APIRouter(prefix="/api/contribute", tags=["contribute"])
URL_PREFIX = "/api/contribute/recordings"


def _project(token: str) -> Project:
    project = find_project_by_token(token)
    if project is None:
        raise HTTPException(403, {"code": "bad_token", "message": "This sharing link is not valid (any more)."})
    return project


def _contributor(name: str) -> str:
    slug = slugify(name)[:24]
    if not slug or slug == "wakeword":
        raise HTTPException(400, {"code": "name_required", "message": "Contributor name is required."})
    return slug


@router.get("/info")
def info(token: str = Query(...)):
    project = _project(token)
    settings = load_settings(project)
    positives = len(list(project.positive_dir.glob("*.wav")))
    return {
        "project": settings["name"],
        "wake_word": settings["wake_word"],
        "max_record_seconds": settings["max_record_seconds"],
        "positive_count": positives,
        "contributor_target": int(settings.get("contributor_target") or 10),
        "tags": list(recordings.TAGS),
    }


@router.get("/recordings")
def list_own(token: str = Query(...), name: str = Query(...), kind: str = "positive"):
    project = _project(token)
    items = recordings.list_recordings(kind, project, contributor=_contributor(name), url_prefix=URL_PREFIX)
    for item in items:
        item["url"] += f"?token={token}"
    return {"kind": kind, "items": items, "count": len(items)}


@router.post("/recordings")
async def upload(token: str = Query(...), name: str = Query(...), file: UploadFile = File(...), kind: str = Form("positive"), tag: str | None = Form(None)):
    project = _project(token)
    item = await recordings.store_upload(file, kind, project, contributor=_contributor(name), url_prefix=URL_PREFIX, tag=tag or None)
    item["url"] += f"?token={token}"
    return item


@router.get("/recordings/{kind}/{rec_id}")
def play(kind: str, rec_id: str, token: str = Query(...)):
    return recordings.file_response(kind, rec_id, _project(token))


@router.delete("/recordings/{kind}/{rec_id}")
def delete(kind: str, rec_id: str, token: str = Query(...), name: str = Query(...)):
    project = _project(token)
    if recordings.contributor_of(rec_id) != _contributor(name):
        raise HTTPException(403, {"code": "not_owner", "message": "You can only delete your own recordings."})
    recordings.soft_delete(kind, rec_id, project)
    return {"deleted": rec_id, "restorable": True}


@router.post("/recordings/{kind}/{rec_id}/restore")
def restore(kind: str, rec_id: str, token: str = Query(...), name: str = Query(...)):
    project = _project(token)
    if recordings.contributor_of(rec_id) != _contributor(name):
        raise HTTPException(403, {"code": "not_owner", "message": "You can only restore your own recordings."})
    item = recordings.restore(kind, rec_id, project, url_prefix=URL_PREFIX)
    item["url"] += f"?token={token}"
    return item
