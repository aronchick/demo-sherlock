from __future__ import annotations

import asyncio
import json
import pathlib
import queue
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..events import BUS
from ..orchestrator import Sherlock

STATIC = pathlib.Path(__file__).parent / "static"


def make_app(sherlock: Sherlock) -> FastAPI:
    app = FastAPI(title="Sherlock")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/events")
    async def events():
        q = BUS.subscribe()

        async def gen():
            while True:
                try:
                    ev = q.get_nowait()
                    yield f"data: {ev.json()}\n\n"
                except queue.Empty:
                    stats = dict(BUS.stats)
                    stats["vlm_calls_avoided"] = max(0, stats["frames"] - stats["vlm_calls"])
                    yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
                    await asyncio.sleep(0.5)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/stream/{which}")
    async def stream(which: str):
        async def gen():
            while True:
                jpg = sherlock.frame_jpeg(which)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.08)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/demo/cover/{camera}/{state}")
    def cover(camera: str, state: str):
        sherlock.cover(camera, state == "on")
        return {"ok": True}

    @app.post("/demo/fail/{n}")
    def fail(n: int):
        sherlock.inject_failure(n)
        return {"ok": True}

    @app.get("/reports")
    def reports():
        return JSONResponse(sherlock.log[-50:])

    @app.get("/cameras")
    def cameras():
        return {"cameras": [c.describe() for c in sherlock.cameras.values()]}

    return app
