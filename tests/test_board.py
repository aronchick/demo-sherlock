"""The board's backend contract: static files and routes, the measured-results card,
the Scout detector /mode reports, the chain metadata on tool events, the head-only
review crop, and the live-only Scout latency."""
import json
import pathlib
import queue
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from sherlock.agents import scout as scout_module
from sherlock.agents.director import RuleDirector
from sherlock.agents.scout import Detection, Scout, head_region
from sherlock.agents.vision import MockAnalyst, MockVerifier, _mock_classify
from sherlock.camera import FakePTZCamera
from sherlock.config import CameraSpec, Config
from sherlock.dashboard import numbers
from sherlock.dashboard.server import STATIC, make_app
from sherlock.events import BUS
from sherlock.orchestrator import Sherlock
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
        assert client.get("/mode").json() == {"mock": True, "director_model": None, "vlm_model": None,
                                               "scout_backend": None, "scout_ms": None, "servers": None}
        assert client.get("/stream/wide/nope").status_code == 404
        assert client.get("/nope.js").status_code == 404


def measured_ids() -> set[str]:
    """Every element numbers.js writes to."""
    return set(re.findall(r"'(numbers?-[a-z-]+)'", (pathlib.Path(STATIC) / "numbers.js").read_text()))


def empty_results(tmp_path, monkeypatch) -> tuple[str, dict]:
    """The board page and /numbers when no measured result exists yet."""
    monkeypatch.setattr(numbers, "ROOT", tmp_path)
    with TestClient(make_app(SimpleNamespace(cfg=Config(), cameras={}))) as client:
        assert client.get("/numbers.js").status_code == 200
        return client.get("/log").text, client.get("/numbers").json()


def test_board_mounts_every_measured_result_as_not_measured(tmp_path, monkeypatch):
    page, data = empty_results(tmp_path, monkeypatch)
    assert data == {"director": None, "ablation": None, "ablation_secondary": None, "scout": None}
    assert '<script src="numbers.js"' in page
    ids = measured_ids()
    assert len(ids) == 15 and all(f'id="{i}"' in page for i in ids), "numbers.js writes to an element the board lacks"
    outputs = re.findall(r'<output id="(number-[a-z-]+)">([^<]*)</output>', page)
    assert len(outputs) == 9 and {text for _, text in outputs} == {"not measured"}
    assert "TBD" not in page


# numbers.js against a stand-in DOM holding the board's ids, fed a /numbers body.
# A heading prints as its parts: [name, detail, runtime]; a hidden element as "hidden".
RUN_NUMBERS_JS = """
const {script, data, ids} = JSON.parse(process.argv[1]);
const shown = {};
const node = () => ({textContent: '', hidden: false, replaceChildren(...kids) { this.parts = kids.map(k => k.textContent); }});
globalThis.document = {getElementById: id => ids.includes(id) ? (shown[id] ??= node()) : null, createElement: node};
globalThis.fetch = async () => ({ok: true, json: async () => data});
globalThis.setInterval = () => 0;
require(script);
setTimeout(() => console.log(JSON.stringify(Object.fromEntries(
  Object.entries(shown).map(([id, el]) => [id, el.hidden ? 'hidden' : el.parts || el.textContent])))), 0);
"""


def run_numbers_js(page: str, data: dict) -> dict:
    arg = {"script": str(pathlib.Path(STATIC) / "numbers.js"), "data": data, "ids": re.findall(r'id="([^"]+)"', page)}
    run = subprocess.run(["node", "-e", RUN_NUMBERS_JS, json.dumps(arg)], capture_output=True, text=True,
                         timeout=30, check=True)
    return json.loads(run.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run numbers.js")


@needs_node
def test_numbers_js_renders_not_measured_from_an_empty_results_dir(tmp_path, monkeypatch):
    page, data = empty_results(tmp_path, monkeypatch)
    shown = run_numbers_js(page, data)
    assert set(shown) == measured_ids()
    assert shown.pop("numbers-status") == "No measured results yet"
    headings = {i: shown.pop(i) for i in ("number-director", "number-ablation", "number-scout")}
    assert headings == {"number-director": ["Director", "", ""], "number-ablation": ["NeMo Gym", "", ""],
                        "number-scout": ["Scout", " · weapon", ""]}
    secondary = [shown.pop(i) for i in ("number-ablation-secondary", "number-ablation-secondary-where")]
    assert secondary == ["hidden", ""], "no second Gym run, no line for it"
    assert set(shown.values()) == {"not measured"} and len(shown) == 9


@needs_node
def test_every_measured_group_names_its_runtime(tmp_path, monkeypatch):
    page, _ = empty_results(tmp_path, monkeypatch)
    director = {"concurrency": 8, "succeeded": 8, "attempted": 8, "p50_ms": 900, "p95_ms": 1500,
                "output_tokens_per_s": 40, "requests_per_s": 2, "prefix_cache_p95_change_pct": None, "file": "b.json"}
    local = {
        "director": {**director, "measured_on": None},  # a bench that recorded no server
        "ablation": {"config": "nemotron_lightning", "tasks": 8, "task_success": 0.875, "tool_call_validity": 1.0,
                     "measured_on": "local · Ollama", "file": "a.json"},
        "scout": {"source": "coco", "split": "val", "boxes": 104, "before_precision": 0.9048, "after_precision": None,
                  "before_recall": 0.1827, "after_recall": None, "measured_on": "local · MPS", "file": "s.json"},
    }
    shown = run_numbers_js(page, local)
    assert shown["number-director"] == ["Director", " · C8 · 8/8", "runtime not recorded"]
    assert shown["number-ablation"] == ["NeMo Gym", " · 8 tasks", "local · Ollama"]
    assert shown["number-scout"] == ["Scout", " · weapon · coco val n=104", "local · MPS"]
    assert shown["number-precision"] == "90.5% → not measured"
    assert shown["number-ablation-secondary"] == "hidden"
    assert "vLLM" not in json.dumps(shown)
    brev = run_numbers_js(page, {"director": {**director, "measured_on": "brev-l40s · vLLM"}, "ablation": None, "scout": None})
    assert brev["number-director"][2] == "brev-l40s · vLLM"
    # Both Gym runs: Brev leads the group, the local run gets the one secondary line.
    both = run_numbers_js(page, {**local, "ablation": {**local["ablation"], "task_success": 1.0, "measured_on": "Brev · vLLM"},
                                 "ablation_secondary": local["ablation"]})
    assert both["number-ablation"] == ["NeMo Gym", " · 8 tasks", "Brev · vLLM"]
    assert both["number-success"] == "100%"
    assert both["number-ablation-secondary-where"] == "local · Ollama"
    assert both["number-ablation-secondary-success"] == "87.5% task success"
    assert both["number-ablation-secondary"] != "hidden"


def scout_backend(tmp_path, *, mock: bool, synthetic: bool, backend: str = "yolo") -> str:
    """What /mode says the Scout runs for this runtime."""
    cfg = Config(mock=mock, scout_backend=backend, cameras=[CameraSpec("hero", "fake", "synthetic")])
    s = Sherlock(cfg, synthetic=synthetic, evidence_dir=str(tmp_path))
    return TestClient(make_app(s)).get("/mode").json()["scout_backend"]


def test_mode_names_the_scout_detector_that_runs(tmp_path, monkeypatch):
    class FakeYolo:  # ultralytics is not installed for mock runs or the test gate
        def __init__(self, model_path, *, conf=0.35):
            pass

    monkeypatch.setattr(scout_module, "YoloScout", FakeYolo)
    monkeypatch.setenv("MODEL_SERVER", "vLLM")  # a live /mode must not probe model servers
    monkeypatch.delenv("SCOUT_MISLABEL", raising=False)
    # Live models on synthetic cameras: the Scout reads scene truth, not YOLO.
    assert scout_backend(tmp_path, mock=False, synthetic=True) == "scene"
    assert scout_backend(tmp_path, mock=False, synthetic=False) == "yolo"
    assert scout_backend(tmp_path, mock=False, synthetic=False, backend="motion") == "motion"
    assert scout_backend(tmp_path, mock=True, synthetic=True) == "scene"
    assert scout_backend(tmp_path, mock=True, synthetic=False) == "motion"  # mock without a scene


# app.js against a stand-in DOM (any unknown property is a callable stub), fed one
# runtime's /mode, /cameras and /reports and then a first stats event.
RUN_APP_JS = r"""
{  // a block: app.js declares its own top-level names in this same global scope
const {script, routes} = JSON.parse(process.argv[1]);
const handler = {
  get(t, k) {
    if (k === Symbol.toPrimitive) return () => '';
    if (k === 'then') return undefined;
    if (!(k in t)) t[k] = stub();
    return t[k];
  },
  apply: () => stub(),
  construct: () => stub(),
};
const stub = (init = {}) => new Proxy(Object.assign(function () {}, init), handler);
const els = {};
const el = id => (els[id] ??= stub({id, textContent: '', title: '', hidden: false, src: '', dataset: {}, disabled: false}));
let stats = null;
Object.assign(globalThis, {
  document: stub({getElementById: el, createElement: () => stub({dataset: {}}), body: stub({dataset: {}})}),
  matchMedia: () => ({matches: false}), requestAnimationFrame: () => 0, setInterval: () => 0,
  devicePixelRatio: 1, getComputedStyle: () => stub(), ResizeObserver: function () { return stub(); },
  EventSource: function () { return stub({addEventListener: (type, fn) => { if (type === 'stats') stats = fn; }}); },
  fetch: async url => {
    const body = routes[url.split('?')[0]];
    return {ok: body !== undefined, status: body === undefined ? 404 : 200, json: async () => body};
  },
});
require('vm').runInThisContext(require('fs').readFileSync(script, 'utf8'), {filename: script});
setTimeout(() => {
  stats({data: JSON.stringify({frames: 0})});
  const tile = c => ({mark: el('cam-' + c).dataset.unconfigured ? 'not in this run'
                        : el('cam-' + c).dataset.unavailable ? 'unavailable' : 'open',
                      stream: el('img-' + c).src.split('?')[0]});
  console.log(JSON.stringify({pill: el('pill').textContent, hero: tile('hero'), scout1: tile('scout1')}));
}, 50);
}
"""


def board_cameras(tmp_path, *, mock: bool, synthetic: bool, cameras: list[str]) -> dict:
    """What the board shows once a runtime with these cameras answers and streams."""
    cfg = Config(mock=mock, cameras=[CameraSpec(name, "fake", "synthetic") for name in cameras])
    client = TestClient(make_app(Sherlock(cfg, synthetic=synthetic, evidence_dir=str(tmp_path))))
    routes = {path: client.get(path).json() for path in ("/mode", "/cameras", "/reports")}
    arg = {"script": str(pathlib.Path(STATIC) / "app.js"), "routes": routes}
    run = subprocess.run(["node", "-e", RUN_APP_JS, json.dumps(arg)], capture_output=True, text=True,
                         timeout=30, check=True)
    return json.loads(run.stdout)


@needs_node
def test_board_counts_only_the_cameras_the_run_opened(tmp_path, monkeypatch):
    class FakeYolo:  # ultralytics is not installed for mock runs or the test gate
        def __init__(self, model_path, *, conf=0.35):
            pass

    monkeypatch.setattr(scout_module, "YoloScout", FakeYolo)
    monkeypatch.setenv("MODEL_SERVER", "vLLM")  # a live /mode must not probe model servers
    monkeypatch.delenv("SCOUT_MISLABEL", raising=False)
    # A one-camera live run: scout1 was never configured, so it is not watched or streamed.
    assert board_cameras(tmp_path, mock=False, synthetic=False, cameras=["hero"]) == {
        "pill": "MONITORING · 1 CAMERA",
        "hero": {"mark": "open", "stream": "/stream/wide/hero"},
        "scout1": {"mark": "not in this run", "stream": ""},
    }
    # The two-camera synthetic run watches both.
    assert board_cameras(tmp_path, mock=True, synthetic=True, cameras=["hero", "scout1"]) == {
        "pill": "MONITORING · 2 CAMERAS",
        "hero": {"mark": "open", "stream": "/stream/wide/hero"},
        "scout1": {"mark": "open", "stream": "/stream/wide/scout1"},
    }


def test_board_labels_every_scout_detector_by_name():
    app_js = (pathlib.Path(STATIC) / "app.js").read_text()
    table = re.search(r"const SCOUT = \{(.*?)\n\};", app_js, re.DOTALL).group(1)
    labels = dict(re.findall(r'^\s+(\w+): \["([^"]+)"', table, re.MULTILINE))
    assert labels == {"scene": "scene truth", "yolo": "YOLO", "motion": "Motion detector"}
    assert "YOLO + motion" not in app_js, "no Scout runs YOLO and motion together"


def test_review_crop_is_the_head_not_the_knife():
    cams, det = knife_scene("hero")
    tools = ToolBox(cams, MockAnalyst(), verifier=MockVerifier())
    tools.begin_incident(det, "t")
    assert "error" not in tools.call("look_at", {"camera": "hero", "bbox": list(det.head)})
    tools.call("snapshot", {"camera": "hero", "subject": "face"})
    face = tools.evidence["face_crop"]
    assert face.shape[0] < 720 and face.shape[1] < 640, "a head crop, not the whole zoomed view"
    assert _mock_classify(face) is None, "no blade in the review crop"
    # The weapon crop the models see is the knife box plus a margin, not the zoomed view.
    tools.call("look_at", {"camera": "hero", "bbox": list(det.bbox)})
    tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    weapon = tools.evidence["weapon_crop"]
    assert weapon.shape[1] < 640, "the box plus a margin, not the whole zoomed view"
    assert _mock_classify(weapon) == "knife"


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
