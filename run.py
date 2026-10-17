"""Run Sherlock.  `python run.py` (real cameras+models per .env)
                 `python run.py --mock --synthetic` (no hardware, no models)"""
import argparse
import os
import sys

import uvicorn

# The package lives at sherlock/sherlock; make `python run.py` work from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sherlock"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="rule-based director + mock VLMs, no model server")
    p.add_argument("--synthetic", action="store_true", help="generated video instead of real cameras")
    p.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8000")))
    a = p.parse_args()
    if a.mock:
        os.environ["SHERLOCK_MOCK"] = "1"
    from sherlock.config import Config
    from sherlock.dashboard.server import make_app
    from sherlock.orchestrator import Sherlock

    s = Sherlock(Config(), synthetic=a.synthetic)
    s.start()
    uvicorn.run(make_app(s), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
