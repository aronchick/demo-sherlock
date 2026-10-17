"""Security weapon flow: weapon -> full evidence chain + alert; benign -> ignore
with zero VLM calls; the face crop never reaches a model."""
import json
import queue
from types import SimpleNamespace

import httpx
import numpy as np
import openai
from sherlock.agents.director import Director, RuleDirector
from sherlock.agents.scout import Detection, Scout, head_region
from sherlock.agents.vision import Analyst, MockAnalyst, MockVerifier, Verifier
from sherlock.camera import CameraError, FakePTZCamera
from sherlock.config import Config, Endpoint
from sherlock.events import BUS
from sherlock.orchestrator import Sherlock
from sherlock.scene import ScriptedScene
from sherlock.tools import TOOL_SCHEMAS, ToolBox
from test_pipeline_mock import is_alert, wait_for

PERSON, KNIFE, CUP = (500, 230, 700, 646), (672, 254, 696, 404), (100, 300, 140, 360)


def drain(q: queue.Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def chain(events: list, incident: str) -> list[str]:
    return [e.data["tool"] for e in events if e.stage == "tool" and e.data.get("incident") == incident]


def spy_models(s: Sherlock) -> list[tuple[str, bool, bool]]:
    """Record every model call as (who, saw_weapon_crop, saw_face_crop)."""
    seen: list[tuple[str, bool, bool]] = []
    analyst, verifier = s.tools.analyst, s.tools.verifier

    def mark(who, img):
        ev = s.tools.evidence
        seen.append((who, img is ev.get("weapon_crop"), img is ev.get("face_crop")))

    class A:
        def describe(self, img, question):
            mark("analyst", img)
            return analyst.describe(img, question)

    class V:
        def check(self, img, claim):
            mark("verifier", img)
            return verifier.check(img, claim)

    s.tools.analyst = A()
    s.tools.verifier = s.verifier = V()
    return seen


# --- end to end ---------------------------------------------------------------
def test_ignore_path_makes_zero_vlm_calls(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    q = BUS.subscribe()
    s.start()
    r = wait_for(s, lambda r: r["action"] == "ignore")
    s.stop()
    assert r, "no ignore report"
    assert r["trigger"]["label"] == "cup" and r["trigger"]["threat"] == "benign"
    assert r["vlm_calls"] == 0
    assert r["evidence"] == [] and "verified" not in r
    assert chain(drain(q), r["incident"]) == ["report"], "benign path must not look, snapshot or verify"
    assert not (tmp_path / r["incident"]).exists()


def test_alert_path_runs_full_chain_and_saves_evidence(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    q = BUS.subscribe()
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r, "no alert"
    assert chain(drain(q), r["incident"]) == ["look_at", "snapshot", "look_at", "snapshot", "verify", "report"]
    assert r["trigger"]["threat"] == "weapon" and r["trigger"]["person"] and r["trigger"]["head"]
    assert r["verified"] is True
    assert r["evidence"] == ["weapon_crop", "face_crop"]
    for name in r["evidence"]:
        assert (tmp_path / r["incident"] / f"{name}.jpg").is_file()
    assert r["vlm_calls"] == 2, "analyst on the weapon crop + verifier; the face crop is never analyzed"
    assert BUS.stats["alerts"] >= 1 and BUS.stats["ignored"] >= 1


def test_models_never_see_the_face_crop(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    seen = spy_models(s)
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r
    assert seen, "no model calls recorded"
    assert all(not face for _, _, face in seen)
    assert all(weapon for who, weapon, _ in seen if who == "verifier"), "verifier must check the weapon crop"


def test_second_camera_does_not_duplicate_the_incident(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    s.start()
    wait_for(s, is_alert)
    s.stop()
    labels = [r["trigger"]["label"] for r in s.log]
    assert labels.count("knife") == 1 and labels.count("cup") == 1


# --- units --------------------------------------------------------------------
def test_scout_weapon_event_carries_nearest_person_and_head():
    scout = Scout("scene", "", dwell=0.0, truth=lambda cam: [("person", PERSON), ("knife", KNIFE), ("cup", CUP)])
    frame = np.zeros((720, 1280, 3), np.uint8)
    scout.step(frame, "hero")  # first sighting starts the track
    fresh = {d.label: d for d in scout.step(frame, "hero")}
    assert set(fresh) == {"knife", "cup"}, "people are context, never their own event"
    knife, cup = fresh["knife"], fresh["cup"]
    assert knife.threat == "weapon" and knife.person == PERSON and knife.head == head_region(PERSON)
    hx1, hy1, hx2, hy2 = knife.head
    assert PERSON[0] <= hx1 < hx2 <= PERSON[2] and hy1 == PERSON[1] and hy2 < PERSON[1] + (PERSON[3] - PERSON[1]) / 3
    assert cup.threat == "benign" and cup.person is None
    assert scout.step(frame, "hero") == [], "one event per object"


def test_motion_scout_fallback_still_emits():
    scout = Scout("mock", "", dwell=0.0)
    blank = np.zeros((360, 640, 3), np.uint8)
    obj = blank.copy()
    obj[100:220, 300:400] = 255
    scout.step(blank, "hero")
    out = scout.step(obj, "hero") + scout.step(obj, "hero")
    assert out and out[0].label == "motion" and out[0].threat == "unknown"


def _toolbox():
    cam = FakePTZCamera("hero", None)
    cam.src = ScriptedScene("hero", speed=0)
    return ToolBox({"hero": cam}, MockAnalyst(), verifier=MockVerifier())


def test_report_schema_and_dispatch_carry_evidence():
    report = next(t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == "report")
    assert report["parameters"]["properties"]["evidence"]["type"] == "array"
    assert {t["function"]["name"] for t in TOOL_SCHEMAS} >= {"look_at", "snapshot", "verify", "report"}
    tools = _toolbox()
    tools.begin_incident(Detection("knife", 0.9, KNIFE, "hero", threat="weapon"), "x")
    out = tools.call("report", {"claim": "a knife", "confidence": 0.9, "action": "alert",
                                "evidence": ["weapon_crop"]})
    assert out["evidence"] == [] and out["evidence_unknown"] == ["weapon_crop"]
    assert "error" in tools.call("report", {"claim": "x", "confidence": 1, "action": "escalate"})


def test_face_snapshot_is_stored_but_never_analyzed():
    tools = _toolbox()
    tools.analyst = SimpleNamespace(describe=lambda *a: (_ for _ in ()).throw(AssertionError("analyst called")))
    tools.begin_incident(Detection("knife", 0.9, KNIFE, "hero", threat="weapon"), "x")
    vlm_before = BUS.stats["vlm_calls"]
    out = tools.call("snapshot", {"camera": "hero", "subject": "face"})
    assert out == {"evidence_id": "face_crop", "stored": True, "analyzed": False}
    assert "face_crop" in tools.evidence and BUS.stats["vlm_calls"] == vlm_before
    assert "error" in tools.call("verify", {"claim": "a knife"}), "nothing but a face to verify against"


def test_failed_face_zoom_still_alerts_with_weapon_crop():
    tools = _toolbox()
    head = head_region(PERSON)
    look_at = tools.t_look_at

    def flaky_look_at(camera, bbox):
        if tuple(bbox) == head:
            raise CameraError("gimbal stalled")
        return look_at(camera, bbox)

    tools.t_look_at = flaky_look_at
    tools.cameras["hero"].src = ScriptedScene("hero", t0=0, speed=0)
    tools.cameras["hero"].src.clock = lambda: 17.0  # knife held up
    det = Detection("knife", 0.9, KNIFE, "hero", threat="weapon", person=PERSON, head=head)
    tools.begin_incident(det, "x")
    out = RuleDirector(tools).investigate(det)
    assert out["action"] == "alert" and out["evidence"] == ["weapon_crop"]


def test_benign_snapshot_is_refused():
    tools = _toolbox()
    tools.begin_incident(Detection("cup", 0.9, CUP, "hero", threat="benign"), "x")
    assert "error" in tools.call("snapshot", {"camera": "hero", "subject": "object"})
    assert tools.evidence == {}


def test_prompts_never_ask_about_identity():
    for prompt in (Analyst.PROMPT, Verifier.PROMPT):
        low = prompt.lower()
        assert "identify" in low and ("never identify" in low or "do not identify" in low)
        for banned in ("who is", "name of the person", "recognize"):
            assert banned not in low


# --- the LLM Director, with a scripted stand-in for the model ------------------
def _call(i, name, args):
    return SimpleNamespace(id=f"c{i}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class _ScriptedLLM:
    def __init__(self, steps):
        self.steps, self.prompts = iter(steps), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.prompts.append(kw["messages"])
        i, (name, args) = next(self.steps)
        msg = SimpleNamespace(content=None, tool_calls=[_call(i, name, args)])
        msg.model_dump = lambda **_: {"role": "assistant", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_llm_director_weapon_chain_reports_evidence():
    tools = _toolbox()
    head = head_region(PERSON)
    det = Detection("knife", 0.9, KNIFE, "hero", threat="weapon", person=PERSON, head=head)
    tools.begin_incident(det, "x")
    d = Director(Endpoint("http://unused/v1", "m"), tools)
    d.client = _ScriptedLLM(enumerate([
        ("look_at", {"camera": "hero", "bbox": list(KNIFE)}),
        ("snapshot", {"camera": "hero", "subject": "weapon", "question": "What object is this?"}),
        ("look_at", {"camera": "hero", "bbox": list(head)}),
        ("snapshot", {"camera": "hero", "subject": "face"}),
        ("verify", {"claim": "a kitchen knife"}),
        ("report", {"claim": "a kitchen knife", "confidence": 0.9, "action": "alert",
                    "evidence": ["weapon_crop", "face_crop"]}),
    ]))
    out = d.investigate(det)
    assert out["action"] == "alert" and out["evidence"] == ["weapon_crop", "face_crop"]
    first_user = json.loads(d.client.prompts[0][1]["content"])
    assert first_user["threat"] == "weapon" and first_user["head_bbox"] == list(head)


def test_unreachable_director_falls_back_to_rules():
    """Guards the deliberately broad except in Director.investigate: any endpoint
    failure (connection error or something odd mid-pitch) must degrade, not crash."""
    for err in (openai.APIConnectionError(request=httpx.Request("POST", "http://unused/v1")),
                RuntimeError("proxy returned HTML")):
        tools = _toolbox()
        det = Detection("cup", 0.9, CUP, "hero", threat="benign")
        tools.begin_incident(det, "x")
        d = Director(Endpoint("http://unused/v1", "m"), tools)

        def boom(*_, err=err, **__):
            raise err

        d.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
        out = d.investigate(det)
        assert out["done"] and out["action"] == "ignore", f"no rule-based fallback for {type(err).__name__}"
