from __future__ import annotations

import asyncio
import json
import pathlib
import queue
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from ..events import BUS
from ..orchestrator import Sherlock
from .model_servers import describe
from .numbers import read_numbers

STATIC = pathlib.Path(__file__).parent / "static"
_INCIDENT_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{3,}$")
_EVIDENCE_NAMES = {"weapon_crop", "face_crop", "object_crop"}


def make_app(sherlock: Sherlock) -> FastAPI:
    app = FastAPI(title="Sherlock")
    # The board's assets (styles.css, app.js, icons.svg) are served from the site
    # root, so its links stay relative and static/ also renders on its own.
    # StaticFiles is the router's fallback, not a mount at "/": a "/" mount would
    # shadow routes added after make_app(), such as the data-flow page at "/".
    app.router.default = StaticFiles(directory=STATIC, html=True)

    @app.get("/log", response_class=HTMLResponse)  # the board; the data-flow page (pipeline.py) is mounted at /
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/numbers")
    def numbers():
        return JSONResponse(read_numbers(), headers={"Cache-Control": "no-store"})

    @app.get("/mode")
    def mode():
        """What the board labels itself with: the mock runtime or the live models,
        and which program serves them (asked of each server, never assumed)."""
        cfg = sherlock.cfg
        return {"mock": cfg.mock,
                "director_model": None if cfg.mock else cfg.director.model,
                "vlm_model": None if cfg.mock else cfg.analyst.model,
                "scout_ms": BUS.snapshot().get("scout_ms"),
                "servers": describe(cfg)}

    @app.get("/events")
    async def events():
        q = BUS.subscribe()

        async def gen():
            while True:
                try:
                    ev = q.get_nowait()
                    yield f"data: {ev.json()}\n\n"
                except queue.Empty:
                    yield f"event: stats\ndata: {json.dumps(BUS.snapshot())}\n\n"
                    await asyncio.sleep(0.5)
        return StreamingResponse(gen(), media_type="text/event-stream")

    def mjpeg(which: str, camera: str | None = None) -> StreamingResponse:
        async def gen():
            while True:
                jpg = sherlock.frame_jpeg(which, camera)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.08)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream/{which}")
    async def stream(which: str):
        return mjpeg(which)  # /stream/wide is the first camera (the /pipeline view uses it)

    @app.get("/stream/wide/{camera}")
    async def stream_camera(camera: str):
        if camera not in sherlock.cameras:
            raise HTTPException(404)
        return mjpeg("wide", camera)

    @app.get("/evidence/{incident}/{name}.jpg")
    def evidence(incident: str, name: str):
        if not _INCIDENT_ID.match(incident) or name not in _EVIDENCE_NAMES:
            raise HTTPException(404)
        path = sherlock.evidence_path(incident, name)
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/demo/cover/{camera}/{state}")
    def cover(camera: str, state: str):
        if not sherlock.cover(camera, state == "on"):
            raise HTTPException(404, f"camera {camera!r} is not available")
        return {"ok": True}

    @app.post("/demo/fail/{n}")
    def fail(n: int):
        sherlock.inject_failure(n)
        return {"ok": True}

    @app.post("/demo/corrupt/{n}")
    def corrupt(n: int):
        sherlock.inject_corruption(n)
        return {"ok": True}

    @app.get("/reports")
    def reports():
        return JSONResponse(sherlock.log[-50:])

    @app.get("/cameras")
    def cameras():
        # The pages read `unavailable` on load: a camera that failed at startup
        # was announced before any browser was listening.
        return {"cameras": [c.describe() for c in sherlock.cameras.values()],
                "unavailable": [{"name": n, "error": e} for n, e in sherlock.unavailable.items()]}

    return app
