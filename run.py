"""Run Sherlock.  `uv run --extra yolo python run.py` (real cameras+models per .env; `just live`)
`uv run python run.py --mock --synthetic` (no hardware, no models; `just demo-up`)"""
import argparse
import os
import sys

import uvicorn

# The package lives at sherlock/sherlock; make `uv run python run.py` work from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sherlock"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="rule-based director + mock VLMs, no model server")
    p.add_argument("--synthetic", action="store_true", help="generated video instead of real cameras")
    p.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8000")))
    a = p.parse_args()
    if a.mock:
        os.environ["SHERLOCK_MOCK"] = "1"
    from sherlock.camera import CameraError
    from sherlock.config import Config
    from sherlock.dashboard.pipeline import mount_pipeline
    from sherlock.dashboard.server import make_app
    from sherlock.orchestrator import Sherlock

    # Config loads .env; an unset or blank host must never expose camera feeds.
    host = os.environ.get("DASHBOARD_HOST", "").strip() or "127.0.0.1"
    try:
        s = Sherlock(Config(), synthetic=a.synthetic)
    except CameraError as e:  # only when no camera opened; one line, not a traceback
        sys.exit(f"Sherlock: {e}")
    s.start()
    app = mount_pipeline(make_app(s))  # /pipeline: data-flow view
    uvicorn.run(app, host=host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
