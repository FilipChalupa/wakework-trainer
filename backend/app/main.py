"""FastAPI application: API routers + static frontend."""
from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import contribute, datasets, livetest, public, recordings, system, training
from .config import DATA_DIR, DEFAULT_TRAINING, current_project, load_settings, save_settings

app = FastAPI(title="Wake Word Trainer", version="1.0.0")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()
APP_USER = os.environ.get("APP_USER", "admin").strip() or "admin"


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    path = request.url.path
    if not APP_PASSWORD or path in ("/api/health", "/contribute") or path.startswith(("/api/contribute/", "/api/public/", "/assets/")):
        return await call_next(request)
    header = request.headers.get("authorization", "")
    ok = False
    if header.lower().startswith("basic "):
        try:
            user, _, password = base64.b64decode(header[6:]).decode().partition(":")
            ok = secrets.compare_digest(user, APP_USER) and secrets.compare_digest(password, APP_PASSWORD)
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="wakeword-trainer"'})
    return await call_next(request)


app.include_router(recordings.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(livetest.router)
app.include_router(contribute.router)
app.include_router(system.router)
app.include_router(public.router)


@app.get("/api/health")
def health():
    return {"ok": True, "data_dir": str(DATA_DIR)}


def _config_payload(project, settings):
    return {"project": {**settings, "id": project.id}, "defaults": DEFAULT_TRAINING}


@app.get("/api/config")
def get_config():
    project = current_project()
    return _config_payload(project, load_settings(project))


@app.put("/api/config")
async def put_config(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"detail": "Expected object"}, status_code=400)
    project = current_project()
    return _config_payload(project, save_settings(project, body))


# ----- static frontend (built by Vite into ../static) ----------------------
STATIC_DIR = Path(os.environ.get("STATIC_DIR", Path(__file__).resolve().parent.parent / "static"))
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
