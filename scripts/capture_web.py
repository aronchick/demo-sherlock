"""Shoot the in-room Scout photos from a browser: live preview, one button per
prop, a countdown you can walk back from, and each frame as it lands.

    uv run scripts/capture_web.py --source "name:Insta360"
    open http://127.0.0.1:8010            (space = start the next planned session)

Writes the same data/props/<prop>/<session>/NNNN.jpg layout as capture_props.py,
through the same capture loop, so prelabel, review and build need nothing new.
Localhost only: the preview shows whoever is in the room."""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from capture_props import capture, open_camera, session_dir
from scout_data import IMAGE_EXTS, PROP_CLASS
from sherlock.camera.base import CameraError

PAGE = Path(__file__).with_name("capture_web.html")
FONTS = ROOT / "sherlock" / "sherlock" / "dashboard" / "static" / "fonts"
DEFAULT_PLAN = ("fedora", "fedora", "cap", "coke", "coke", "none")
PREVIEW_WIDTH = 960  # the preview only needs to show framing; saved frames stay full size


class Start(BaseModel):
    prop: str


class Recorder:
    """One session at a time on a background thread; the page polls status()."""

    def __init__(self, cam, out: Path, camera: str, plan: list[str], n: int, interval: float,
                 countdown: float):
        self.cam, self.out, self.camera, self.plan = cam, out, camera, plan
        self.n, self.interval, self.countdown = n, interval, countdown
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.state, self.prop, self.saved, self.go_at = "idle", None, 0, 0.0
        self.session: Path | None = None
        self.recent: list[Path] = []
        self.done: list[str] = []  # props finished this run, to tick off the plan
        self.error: str | None = None

    def start(self, prop: str) -> None:
        if prop not in PROP_CLASS:
            raise HTTPException(400, f"unknown prop {prop!r}")
        with self.lock:
            if self.state != "idle":
                raise HTTPException(409, f"already {self.state}")
            self.stop.clear()
            self.state, self.prop, self.saved, self.error = "countdown", prop, 0, None
            self.go_at = time.monotonic() + self.countdown
            self.session = session_dir(self.out, prop, self.camera)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            if self.stop.wait(max(0.0, self.go_at - time.monotonic())):
                return
            with self.lock:
                self.state = "capturing"
            capture(self.cam, self.session, self.n, self.interval, on_frame=self._saved, stop=self.stop)
            with self.lock:
                if self.saved:
                    self.done.append(self.prop)
        except (CameraError, OSError, SystemExit, cv2.error) as e:  # shown on the page; server keeps running
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.state = "idle"

    def _saved(self, path: Path) -> None:
        with self.lock:
            self.saved += 1
            self.recent = [path, *self.recent][:8]

    def shutdown(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)

    def counts(self) -> dict[str, dict[str, int]]:
        out = {}
        for prop in PROP_CLASS:
            sessions = [d for d in (self.out / prop).glob("*") if d.is_dir()]
            frames = sum(1 for d in sessions for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            out[prop] = {"sessions": len(sessions), "frames": frames}
        return out

    def status(self) -> dict:
        with self.lock:
            left = max(0.0, self.go_at - time.monotonic()) if self.state == "countdown" else 0.0
            done = list(self.done)
            plan = []
            for prop in self.plan:  # each finished session ticks the first matching plan entry
                hit = prop in done
                if hit:
                    done.remove(prop)
                plan.append({"prop": prop, "done": hit})
            return {
                "state": self.state, "prop": self.prop, "saved": self.saved, "n": self.n,
                "countdown_left": round(left, 1), "camera": self.camera, "error": self.error,
                "session": str(self.session.relative_to(self.out)) if self.session else None,
                "recent": [f"/frames/{p.relative_to(self.out).as_posix()}" for p in self.recent],
                "plan": plan, "props": sorted(PROP_CLASS), "counts": self.counts(),
            }


def make_app(rec: Recorder) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        rec.shutdown()
        rec.cam.close()  # stop the frame reader, or Python segfaults on exit (macOS)

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/fonts/{name}")
    def font(name: str) -> FileResponse:
        path = FONTS / name
        if path.parent != FONTS or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/status")
    def status() -> dict:
        return rec.status()

    @app.post("/api/start")
    def start(body: Start) -> dict:
        rec.start(body.prop)
        return rec.status()

    @app.post("/api/stop")
    def stop() -> dict:
        rec.stop.set()
        return rec.status()

    @app.get("/frames/{rel:path}")
    def frame(rel: str) -> FileResponse:
        path = (rec.out / rel).resolve()
        if not path.is_relative_to(rec.out.resolve()) or path.suffix.lower() not in IMAGE_EXTS \
                or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            while True:
                f = await asyncio.to_thread(rec.cam.wide_frame)
                h, w = f.shape[:2]
                if w > PREVIEW_WIDTH:
                    f = cv2.resize(f, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)))
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                await asyncio.sleep(0.08)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="hero", help="camera name from CAMERAS; also names the session")
    p.add_argument("--source", help="override: device index, 'name:<camera>', file or 'synthetic'")
    p.add_argument("--plan", nargs="+", default=list(DEFAULT_PLAN), choices=sorted(PROP_CLASS),
                   help="sessions to tick off, in order")
    p.add_argument("--n", type=int, default=40, help="frames per session")
    p.add_argument("--interval", type=float, default=0.7, help="seconds between frames")
    p.add_argument("--countdown", type=float, default=5.0, help="seconds to walk into frame")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "props")
    p.add_argument("--port", type=int, default=8010)
    a = p.parse_args(argv)
    rec = Recorder(open_camera(a.camera, a.source), a.out, a.camera, a.plan, a.n, a.interval, a.countdown)
    print(f"capture page: http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    uvicorn.run(make_app(rec), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
