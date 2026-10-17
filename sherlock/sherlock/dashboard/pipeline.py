"""Edge-to-cloud pipeline view: a second judge-facing page that animates each
investigation as it crosses the laptop/cloud boundary. It only consumes the
existing /events SSE stream and /stream MJPEG feeds; nothing here touches the
agents, the orchestrator or the main dashboard."""
from __future__ import annotations

import pathlib

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

STATIC = pathlib.Path(__file__).parent / "static"

router = APIRouter()


@router.get("/pipeline", response_class=HTMLResponse)
def pipeline_page() -> str:
    return (STATIC / "pipeline.html").read_text()


def mount_pipeline(app: FastAPI) -> FastAPI:
    """Attach the pipeline page to an app built by make_app(). Mounts in place and
    returns the same object, so run.py can hand uvicorn exactly what make_app built
    (tests stub make_app with a bare object and assert on that identity)."""
    if hasattr(app, "include_router"):
        app.include_router(router)
    return app
