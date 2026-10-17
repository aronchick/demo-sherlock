"""Data-flow view (the default page) and the topology page. Both only consume the
existing /events SSE stream and /stream MJPEG feeds; nothing here touches the
agents, the orchestrator or the event log page."""
from __future__ import annotations

import json
import os
import pathlib

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ..config import Config
from .model_servers import describe, where

STATIC = pathlib.Path(__file__).parent / "static"
FONTS = STATIC / "fonts"

router = APIRouter()


def page_config() -> dict:
    """What the pages may claim, read from the environment on every request.

    mock: the models are in-process mocks, so nothing leaves the laptop.
    expanso: SHERLOCK_EXPANSO=connected, set only once an Expanso Edge pipeline is
    really processing this data. The footer shows "not connected" until then.
    models: the configured model per role and where its endpoint points. A
    localhost endpoint is reported as the laptop (an SSH tunnel to Brev also looks
    like that, so BREV_HOST names the box when a tunnel is in use).
    servers: which program serves each role (vLLM, Ollama, mlx-vlm), asked of the
    server itself; None in a mock run. The pages name a runtime only from this.
    """
    cfg = Config()
    endpoints = {
        "director": cfg.director,
        "analyst": cfg.analyst,
        "verifier": cfg.verifier,
    }
    models = {
        "scout": {"model": cfg.yolo_model if cfg.scout_backend == "yolo" else cfg.scout_backend, "where": "laptop"},
    }
    for role, ep in endpoints.items():
        models[role] = {"model": ep.model, "where": where(ep.base_url), "url": ep.base_url}
    return {
        "mock": cfg.mock,
        "expanso": os.environ.get("SHERLOCK_EXPANSO") == "connected",
        "cameras": [{"name": c.name, "backend": c.backend} for c in cfg.cameras],
        "models": models,
        "servers": describe(cfg),
    }


def _render(name: str) -> str:
    return (STATIC / name).read_text().replace("__CFG_JSON__", json.dumps(page_config()))


@router.get("/", response_class=HTMLResponse)
def flow_page() -> str:
    return _render("pipeline.html")


@router.get("/pipeline", response_class=HTMLResponse)
def pipeline_page() -> str:  # kept so older links and bookmarks still work
    return _render("pipeline.html")


@router.get("/topology", response_class=HTMLResponse)
def topology_page() -> str:
    return _render("topology.html")


@router.get("/pipeline/fonts/{name}")
def pipeline_font(name: str) -> FileResponse:
    """Serve a vendored font by exact file name; anything not in the folder is a 404."""
    if name not in {p.name for p in FONTS.glob("*.woff2")}:
        raise HTTPException(status_code=404)
    return FileResponse(FONTS / name, media_type="font/woff2", headers={"Cache-Control": "public, max-age=86400"})


def mount_pipeline(app: FastAPI) -> FastAPI:
    """Attach the flow and topology pages to an app built by make_app()."""
    app.include_router(router)
    return app
