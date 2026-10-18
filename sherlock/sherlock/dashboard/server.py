from __future__ import annotations

import asyncio
import json
import os
import pathlib
import queue
import re
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from ..events import BUS
from ..orchestrator import Sherlock
from ..receipts import ReceiptStore
from .model_servers import describe
from .numbers import read_numbers

STATIC = pathlib.Path(__file__).parent / "static"
_INCIDENT_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{3,}$")
_EVIDENCE_NAMES = {"model_context", "original_frame", "weapon_crop", "face_crop", "object_crop"}
_SERVING_SERVERS = {"director", "vlm"}
_SERVING_FIELDS = {"up", "at", "kv_cache_usage", "prefix_cache_queries", "prefix_cache_hits",
                   "requests_running", "requests_waiting", "error"}
_SERVING_STALE_S = 60.0
_LOOPBACK = {"127.0.0.1", "::1"}


def expanso_mode() -> str | None:
    """local or cloud, as written by scripts/expanso.sh while its nodes run here."""
    path = pathlib.Path(os.environ.get("SHERLOCK_RUNTIME_DIR", ".runtime")) / "expanso" / "mode"
    try:
        mode = path.read_text().strip()
    except OSError:
        return None
    return mode if mode in {"local", "cloud"} else None


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
        which program serves them (asked of each server, never assumed), which
        Scout detector runs (scene truth on a synthetic scene, YOLO or motion), and
        whether a scripted synthetic scene is behind the cameras (--synthetic)."""
        cfg = sherlock.cfg
        return {"mock": cfg.mock,
                "director_model": None if cfg.mock else cfg.director.model,
                "vlm_model": None if cfg.mock else cfg.analyst.model,
                "scout_backend": getattr(getattr(sherlock, "scout", None), "kind", None),
                "synthetic": bool(getattr(sherlock, "scenes", None)),
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

    @app.get("/evidence/{incident}/{name}.png")
    @app.get("/evidence/{incident}/{name}.jpg")
    def evidence(incident: str, name: str):
        if not _INCIDENT_ID.match(incident) or (name not in _EVIDENCE_NAMES and not re.fullmatch(r"[a-f0-9]{64}", name)):
            raise HTTPException(404)
        try:
            if re.fullmatch(r"[a-f0-9]{64}", name):
                path = sherlock.incidents.path(incident, name + ".png")
                if not path.exists():
                    path = sherlock.incidents.path(incident, name + ".jpg")
            else:
                path = sherlock.evidence_path(incident, name)
        except ValueError:
            raise HTTPException(404) from None
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/png" if path.suffix == ".png" else "image/jpeg")

    @app.post("/demo/cover/{camera}/{state}")
    def cover(camera: str, state: str):
        if not sherlock.cover(camera, state == "on"):
            raise HTTPException(404, f"camera {camera!r} is not available")
        return {"ok": True}

    @app.post("/demo/disagree/{n}")
    def disagree(n: int):
        if not sherlock.cfg.mock:
            raise HTTPException(409, "Disagreement injection is available only in mock mode")
        if n != 1:
            raise HTTPException(400, "Inject one investigation at a time")
        sherlock.tools.injected_disagreements = 1
        BUS.emit("system", "Next investigated pair will include a SIMULATED analyst disagreement")
        return {"ok": True, "simulated": True}

    @app.post("/demo/fail/{n}")
    def fail(n: int):
        sherlock.inject_failure(n)
        return {"ok": True}

    @app.post("/demo/corrupt/{n}")
    def corrupt(n: int):
        sherlock.inject_corruption(n)
        return {"ok": True}

    serving: dict[str, dict] = {}
    receipts = ReceiptStore(sherlock.incidents) if hasattr(sherlock, "incidents") else None

    @app.get("/expanso")
    def expanso():
        """What the footers may claim. 'edge' is true only while the Edge pipelines on
        this laptop are accepting this run's events; serving is what the cloud core's
        vLLM metrics pipeline last pushed, dropped once it goes stale."""
        status = sherlock.edge.status() if sherlock.edge else {"edge": False}
        now = time.time()
        fresh = {k: v for k, v in serving.items() if now - v["received"] < _SERVING_STALE_S}
        return JSONResponse({**status, "mode": expanso_mode(), "serving": fresh,
                             "receipts": receipts.list() if receipts else []},
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/receipts")
    async def receipt_in(request: Request):
        if (request.client.host if request.client else "") not in _LOOPBACK:
            raise HTTPException(403)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                raise HTTPException(413)
        if receipts is None:
            raise HTTPException(503)
        try:
            accepted = receipts.accept(json.loads(body))
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise HTTPException(400, "Receipt does not match local signed evidence") from error
        return {"ok": True, "incident": accepted["incident"]}

    @app.post("/api/serving")
    async def serving_in(request: Request):
        """Records from jobs/sherlock-vllm-metrics.yaml, arriving through the SSH
        tunnel's reverse port, so only loopback callers are accepted."""
        if (request.client.host if request.client else "") not in _LOOPBACK:
            raise HTTPException(403)
        body = await request.body()
        if len(body) > 8192:
            raise HTTPException(413)
        try:
            rec = json.loads(body)
        except ValueError:
            raise HTTPException(400) from None
        if not isinstance(rec, dict) or rec.get("server") not in _SERVING_SERVERS:
            raise HTTPException(400)
        serving[rec["server"]] = {**{k: rec[k] for k in _SERVING_FIELDS if k in rec}, "received": time.time()}
        return {"ok": True}

    @app.get("/incidents")
    def incidents(limit: int = Query(default=20, ge=1, le=100),
                  offset: int = Query(default=0, ge=0, le=100000),
                  action: Literal["alert", "log", "ignore"] | None = None,
                  evidence_only: bool = False):
        return JSONResponse(sherlock.incidents.list(limit, offset, action, evidence_only),
                            headers={"Cache-Control": "no-store"})

    @app.get("/incidents/{incident}")
    def incident_record(incident: str):
        try:
            record = sherlock.incidents.read(incident)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(404, "Incident record unavailable") from None
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    @app.post("/incidents/{incident}/acknowledge")
    def acknowledge(incident: str):
        try:
            return sherlock.incidents.acknowledge(incident)
        except FileNotFoundError:
            raise HTTPException(404) from None
        except ValueError:
            raise HTTPException(409, "No simulated notification to acknowledge") from None
        except OSError:
            raise HTTPException(503, "Acknowledgment could not be saved") from None

    @app.get("/incidents/{incident}/signed-history")
    def signed_history(incident: str):
        try:
            return JSONResponse(sherlock.incidents.signed_history(incident),
                                headers={"Cache-Control": "no-store"})
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(404, "Signed history unavailable") from None

    @app.post("/incidents/{incident}/respond")
    async def respond(incident: str, request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("invalid response")
            return sherlock.incidents.respond(incident, body.get("action"))
        except FileNotFoundError:
            raise HTTPException(404, "Incident unavailable") from None
        except (ValueError, TypeError):
            raise HTTPException(409, "Response is unavailable for this incident state or integrity") from None
        except OSError:
            raise HTTPException(503, "Response could not be saved") from None

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
