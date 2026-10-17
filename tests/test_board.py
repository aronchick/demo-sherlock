"""The board's backend contract: static files and routes, the chain metadata on
tool events, the head-only review crop, and the live-only Scout latency."""
import queue
import time
from types import SimpleNamespace

import numpy as np
from fastapi import APIRouter
from fastapi.testclient import TestClient
from sherlock.agents.director import RuleDirector
from sherlock.agents.scout import Detection, Scout, head_region
from sherlock.agents.vision import MockAnalyst, MockVerifier, _mock_classify
from sherlock.camera import FakePTZCamera
from sherlock.config import Config
from sherlock.dashboard.server import make_app
from sherlock.events import BUS
from sherlock.scene import ScriptedScene
from sherlock.tools import ToolBox


def drain(q: queue.Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def knife_scene(*cameras: str) -> tuple[dict, Detection]:
    """Cameras frozen at the moment the knife is held up (t = 17 s of the script)."""
    cams = {}
    for name in cameras:
        cam = FakePTZCamera(name, None)
        cam.src = ScriptedScene(name, t0=time.time() - 17, speed=1)
        cams[name] = cam
    first = cams[cameras[0]]
    first.wide_frame()
    truth = dict(first.src.objects())
    person, knife = truth["person"], truth["knife"]
    det = Detection("knife", 0.9, knife, cameras[0], "weapon", person, head_region(person))
    return cams, det


def test_board_is_served_at_log_and_later_routes_still_win():
    s = SimpleNamespace(cfg=Config(), cameras={"hero": None})
    app = make_app(s)
    later = APIRouter()

    @later.get("/")
    def flow():
        return {"page": "flow"}

    app.include_router(later)  # what the data-flow page does after make_app()
    with TestClient(app) as client:
        page = client.get("/log")
        assert page.status_code == 200 and 'id="avoided"' in page.text
        assert 'href="styles.css"' in page.text and 'src="app.js"' in page.text
        for asset in ("styles.css", "app.js", "icons.svg", "numbers.js"):
            assert client.get(f"/{asset}").status_code == 200
        assert client.get("/").json() == {"page": "flow"}  # beats the static index.html
        assert client.get("/mode").json() == {"mock": True, "director_model": None,
                                               "vlm_model": None, "scout_ms": None, "servers": None}
        assert client.get("/stream/wide/nope").status_code == 404
        assert client.get("/nope.js").status_code == 404


def test_review_crop_is_the_head_not_the_knife():
    cams, det = knife_scene("hero")
    tools = ToolBox(cams, MockAnalyst(), verifier=MockVerifier())
    tools.begin_incident(det, "t")
    assert "error" not in tools.call("look_at", {"camera": "hero", "bbox": list(det.head)})
    tools.call("snapshot", {"camera": "hero", "subject": "face"})
    face = tools.evidence["face_crop"]
    assert face.shape[0] < 720 and face.shape[1] < 640, "a head crop, not the whole zoomed view"
    assert _mock_classify(face) is None, "no blade in the review crop"
    # The weapon crop the models see is unchanged: still the zoomed view with the blade.
    tools.call("look_at", {"camera": "hero", "bbox": list(det.bbox)})
    tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert tools.evidence["weapon_crop"].shape[:2] == (720, 1280)
    assert _mock_classify(tools.evidence["weapon_crop"]) == "knife"


def test_tool_events_carry_the_chain_and_a_plain_director_line():
    cams, det = knife_scene("hero", "scout1")
    tools = ToolBox(cams, MockAnalyst(), verifier=MockVerifier())
    tools.begin_incident(det, "chain")
    q = BUS.subscribe()
    report = RuleDirector(tools).investigate(det)
    events = [e for e in drain(q) if e.data.get("incident") == "chain"]
    assert report["action"] == "alert"
    steps = [(e.data["step"], e.data["subject"], e.data["camera"]) for e in events if e.stage == "tool"]
    assert steps == [(1, "weapon", "hero"), (2, "weapon", "hero"), (3, "face", "hero"),
                     (4, "face", "hero"), (5, None, "hero"), (6, None, None)]
    said = [e.msg for e in events if e.stage == "narration"]
    assert said[0] == "zooming hero to the weapon" and said[-1] == "reporting · alert"
    assert all(e.data["label"] == "knife" and e.data["retry"] is False for e in events)


def test_a_failed_step_retried_elsewhere_is_one_recovery():
    cams, det = knife_scene("hero", "scout1")
    tools = ToolBox(cams, MockAnalyst(), verifier=MockVerifier())
    tools.begin_incident(det, "fault")
    tools.inject_failure(1)
    q = BUS.subscribe()
    before = dict(BUS.stats)
    RuleDirector(tools).investigate(det)
    tool = [e.data for e in drain(q) if e.stage == "tool" and e.data.get("incident") == "fault"]
    assert tool[0]["error"] is True and (tool[0]["step"], tool[0]["camera"]) == (1, "hero")
    assert (tool[1]["step"], tool[1]["camera"], tool[1]["retry"]) == (1, "scout1", True)
    assert not any(t["retry"] for t in tool[2:]), "only the retried step is a retry"
    assert BUS.stats["tool_errors"] - before["tool_errors"] == 1
    assert BUS.stats["recoveries"] - before["recoveries"] == 1
    assert BUS.stats["tool_calls"] - before["tool_calls"] == 7


def test_scout_latency_is_reported_only_when_measured():
    frame = np.zeros((72, 128, 3), np.uint8)
    BUS.scout_samples.clear()
    Scout("motion", "", 0.3).step(frame, "hero")
    assert "scout_ms" not in BUS.snapshot()
    Scout("motion", "", 0.3, measure=True).step(frame, "hero")
    assert BUS.snapshot()["scout_ms"] >= 0
    BUS.scout_samples.clear()


def test_second_camera_sighting_says_who_handled_it():
    boxes = {"hero": [("knife", (10, 10, 20, 40))], "scout1": [("knife", (10, 10, 20, 40))]}
    scout = Scout("scene", "", 0.0, truth=boxes.__getitem__,
                  handled_via=lambda d: "hero" if d.camera == "scout1" else None)
    q = BUS.subscribe()
    frame = np.zeros((72, 128, 3), np.uint8)
    for _ in range(2):
        for cam in ("hero", "scout1"):
            scout.step(frame, cam)
    seen = {e.data["camera"]: e.data.get("handled_via") for e in drain(q) if e.stage == "scout"}
    assert seen == {"hero": None, "scout1": "hero"}
