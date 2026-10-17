"""Data-flow view: a second judge-facing page that shows each investigation as it
moves from the camera through the Scout, the models and out to a decision. It only
consumes the existing /events SSE stream and /stream MJPEG feeds; nothing here
touches the agents, the orchestrator or the main dashboard."""
from __future__ import annotations

import json
import os
import pathlib

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

STATIC = pathlib.Path(__file__).parent / "static"
FONTS = STATIC / "fonts"

router = APIRouter()


def page_config() -> dict[str, bool]:
    """What the page may claim, read from the environment on every request.

    mock: the models are in-process mocks, so nothing leaves the laptop.
    expanso: SHERLOCK_EXPANSO=connected, set only once an Expanso Edge pipeline is
    really processing this data. The footer shows "not connected" until then.
    """
    return {
        "mock": os.environ.get("SHERLOCK_MOCK") == "1",
        "expanso": os.environ.get("SHERLOCK_EXPANSO") == "connected",
    }


@router.get("/pipeline", response_class=HTMLResponse)
def pipeline_page() -> str:
    return (STATIC / "pipeline.html").read_text().replace("__CFG_JSON__", json.dumps(page_config()))


@router.get("/pipeline/fonts/{name}")
def pipeline_font(name: str) -> FileResponse:
    """Serve a vendored font by exact file name; anything not in the folder is a 404."""
    if name not in {p.name for p in FONTS.glob("*.woff2")}:
        raise HTTPException(status_code=404)
    return FileResponse(FONTS / name, media_type="font/woff2", headers={"Cache-Control": "public, max-age=86400"})


def mount_pipeline(app: FastAPI) -> FastAPI:
    """Attach the pipeline page to an app built by make_app()."""
    app.include_router(router)
    return app
