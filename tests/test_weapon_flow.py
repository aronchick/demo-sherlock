"""Security weapon flow: weapon -> full evidence chain + alert; benign -> ignore
with zero VLM calls; the face crop never reaches a model."""
import json
import queue
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
from sherlock.agents.director import Director, RuleDirector, names_claim
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
        seen.append((who, np.array_equal(img, ev.get("model_context")), np.array_equal(img, ev.get("face_crop"))))

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
    assert not list((tmp_path / r["incident"]).glob("*.jpg")), "benign incidents retain metadata only"
    assert s.incidents.read(r["incident"])["notification"]["status"] == "not_sent"


def test_alert_path_runs_full_chain_and_saves_evidence(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    q = BUS.subscribe()
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r, "no alert"
    assert chain(drain(q), r["incident"]) == ["look_at", "snapshot", "look_at", "snapshot", "report"]
    assert r["trigger"]["threat"] == "weapon" and r["trigger"]["person"] and r["trigger"]["head"]
    assert r["verified"] is True
    assert r["evidence"] == ["original_frame", "face_crop", "weapon_crop", "model_context"]
    for name in r["evidence"]:
        assert (tmp_path / r["incident"] / f"{name}.{'png' if name == 'model_context' else 'jpg'}").is_file()
    assert r["vlm_calls"] == 2, "analyst on the weapon crop + verifier; the face crop is never analyzed"
    assert BUS.stats["alerts"] >= 1 and BUS.stats["ignored"] >= 1


def test_rejected_weapon_claim_retains_linked_face_and_scene(tmp_path):
    s = _frozen_sherlock(tmp_path)
    s.tools.verifier = SimpleNamespace(check=lambda img, claim: {
        "supported": False, "confidence": 0.2, "reason": "no weapon"})
    s._investigate(_knife_det())
    r = s.log[-1]
    assert r["verified"] is False and r["action"] == "log"
    assert set(r["evidence"]) == {"original_frame", "weapon_crop", "face_crop", "model_context"}
    assert r["comparison"]["status"] == "disagreed"
    record = s.incidents.read(r["incident"])
    assert all(a["integrity"] == "verified" for a in record["artifacts"])
    for artifact in record["artifacts"]:
        if artifact["name"] in ("weapon_crop", "face_crop"):
            assert artifact["source_artifact"] == "original_frame"
            assert artifact["relationship"] == "subsequent_capture"
            assert artifact["bbox"]
    assert record["notification"]["status"] == "not_sent"


def test_models_share_scene_object_and_face_context(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    seen = spy_models(s)
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r
    assert seen, "no model calls recorded"
    assert all(not face for _, _, face in seen)
    assert all(weapon for who, weapon, _ in seen if who == "verifier"), "verifier must check the complete context"


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


def test_director_event_names_the_claim_to_verify():
    """Nemotron claimed "a knife" for both Scout "scissors" events on 2026-10-17, copying the
    prompt's example. The event now carries the claim; benign events stay unchanged."""
    for label, threat, claim in (("scissors", "weapon", "a pair of scissors"), ("knife", "weapon", "a knife"),
                                 ("cup", "benign", None)):
        tools = _toolbox()
        det = Detection(label, 0.9, CUP, "hero", threat=threat)
        tools.begin_incident(det, "x")
        d = Director(Endpoint("http://unused/v1", "m"), tools)
        d.client = _ScriptedLLM(enumerate([("report", {"claim": "x", "confidence": 0.5, "action": "log"})]))
        d.investigate(det)
        event = json.loads(d.client.prompts[0][1]["content"])
        assert event.get("claim") == claim and ("claim" in event) == (threat == "weapon")


def test_director_turns_thinking_off_for_vllm_and_ollama():
    """Ollama ignores chat_template_kwargs; only reasoning_effort="none" stops Nemotron
    thinking there, and on 2026-10-17 the thinking ate all 300 tokens with no tool call."""
    det = Detection("cup", 0.9, CUP, "hero", threat="benign")
    for thinking, effort in ((False, "none"), (True, None)):
        tools = _toolbox()
        tools.begin_incident(det, "x")
        d = Director(Endpoint("http://unused/v1", "m"), tools, thinking=thinking)
        bodies = []
        llm = _ScriptedLLM(enumerate([("report", {"claim": "a cup", "confidence": 0.9, "action": "ignore"})]))

        def create(llm=llm, bodies=bodies, **kw):
            bodies.append(kw["extra_body"])
            return llm.create(**kw)

        d.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        assert d.investigate(det)["action"] == "ignore"
        assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": thinking}  # vLLM, llama.cpp
        assert bodies[0].get("reasoning_effort") == effort  # Ollama; vLLM maps "none" the same way


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


# --- review fixes: face guard, bound verdicts, weapon claims, evidence writes ---
def _knife_det():
    return Detection("knife", 0.9, KNIFE, "hero", threat="weapon", person=PERSON, head=head_region(PERSON))


def _knife_raised(src):
    src.clock = lambda: 17.0  # the knife is held up at this point in the script
    return src


def _frozen_sherlock(evidence_dir) -> Sherlock:
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(evidence_dir))
    for scene in s.scenes.values():
        _knife_raised(scene)
    return s


def test_head_zoom_never_reaches_a_model_whatever_the_subject():
    tools = _toolbox()
    _knife_raised(tools.cameras["hero"].src)
    seen = []
    tools.analyst = SimpleNamespace(describe=lambda img, q: seen.append("analyst") or {"description": "x"})
    tools.verifier = SimpleNamespace(check=lambda img, c: seen.append("verifier") or {"supported": True})
    tools.begin_incident(_knife_det(), "x")
    assert "error" in tools.call("snapshot", {"camera": "hero", "subject": "weapon"}), "no aim yet"
    tools.call("look_at", {"camera": "hero", "bbox": list(head_region(PERSON))})
    for args in ({"camera": "hero"}, {"camera": "hero", "subject": "weapon"},
                 {"camera": "hero", "subject": "object"}):
        assert "error" in tools.call("snapshot", args), f"head zoom reached a model via {args}"
    tools.call("move", {"camera": "hero", "pan": 10})
    assert "error" in tools.call("snapshot", {"camera": "hero", "subject": "weapon"}), "aim unknown after move"
    assert "error" in tools.call("verify", {"claim": "a knife"})
    assert seen == [] and not {"weapon_crop", "object_crop"} & set(tools.evidence)
    tools.call("look_at", {"camera": "hero", "bbox": list(KNIFE)})
    assert "error" not in tools.call("snapshot", {"camera": "hero"})
    assert seen == ["analyst"], "aimed back at the weapon, the Analyst gets its crop"


def test_face_zoom_left_over_from_the_last_incident_never_reaches_a_model():
    """Incident 1 ends with the camera on the face. Incident 2 has no head box (the
    person detector missed): until a successful look_at, nothing may go to a model."""
    tools = _toolbox()
    _knife_raised(tools.cameras["hero"].src)
    tools.begin_incident(_knife_det(), "first")
    tools.call("look_at", {"camera": "hero", "bbox": list(head_region(PERSON))})
    assert tools.call("snapshot", {"camera": "hero", "subject": "face"})["analyzed"] is False
    seen = []
    tools.analyst = SimpleNamespace(describe=lambda img, q: seen.append("analyst") or {"description": "x"})
    tools.verifier = SimpleNamespace(check=lambda img, c: seen.append("verifier") or {"supported": True})
    tools.begin_incident(Detection("knife", 0.9, KNIFE, "hero", threat="weapon"), "second")
    for args in ({"camera": "hero"}, {"camera": "hero", "subject": "weapon"},
                 {"camera": "hero", "subject": "object"}):
        assert "error" in tools.call("snapshot", args), f"leftover face view reached a model via {args}"
    assert "error" in tools.call("verify", {"claim": "a knife"})
    assert seen == [] and tools.evidence == {}
    tools.call("look_at", {"camera": "hero", "bbox": list(KNIFE)})
    assert "error" not in tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert seen == ["analyst"]


def test_verdict_is_bound_to_claim_and_crop():
    tools = _toolbox()
    _knife_raised(tools.cameras["hero"].src)
    tools.begin_incident(_knife_det(), "x")
    tools.call("look_at", {"camera": "hero", "bbox": list(KNIFE)})
    tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    tools.call("verify", {"claim": "a knife"})
    assert tools.verdict_for("A knife.") is not None, "same claim, same crop"
    assert tools.verdict_for("a cup") is None, "changed claim"
    tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert tools.verdict_for("a knife") is None, "replaced crop"


def test_final_report_rechecks_a_changed_claim_and_a_replaced_crop(tmp_path):
    checked = []

    def check(img, claim):
        checked.append((claim, img))
        ok = "cup" in claim or "knife" in claim and len(checked) > 1
        return {"supported": ok, "confidence": 0.9, "reason": "stub"}

    s = _frozen_sherlock(tmp_path)
    s.tools.verifier = SimpleNamespace(check=check)
    t = s.tools
    look = {"camera": "hero", "bbox": list(KNIFE)}
    knife_alert = {"claim": "a kitchen knife", "confidence": 0.9, "action": "alert"}

    def claim_swap(det):  # verifies a cup, then reports a knife
        t.call("look_at", look)
        t.call("snapshot", {"camera": "hero", "subject": "weapon"})
        t.call("verify", {"claim": "a white cup"})
        return t.call("report", knife_alert)

    s.director = SimpleNamespace(investigate=claim_swap)
    s._investigate(_knife_det())
    r = s.log[-1]
    assert [c for c, _ in checked] == ["a knife"]
    assert r["verified"] is False and r["action"] == "log", "a cup verdict must not verify a knife alert"

    def crop_swap(det):  # verifies the knife, then replaces the crop before reporting
        t.call("look_at", look)
        t.call("snapshot", {"camera": "hero", "subject": "weapon"})
        t.call("verify", {"claim": "a kitchen knife"})
        t.call("snapshot", {"camera": "hero", "subject": "weapon"})
        return t.call("report", knife_alert)

    s.director = SimpleNamespace(investigate=crop_swap)
    s._investigate(_knife_det())
    assert len(checked) == 2
    assert np.array_equal(checked[1][1], t.evidence["model_context"])
    assert checked[1][1] is not checked[0][1]
    assert s.log[-1]["verified"] is True and s.log[-1]["action"] == "alert"


def test_weapon_tag_on_a_benign_object_does_not_alert():
    """YOLO calls a mug a knife; the Analyst correctly sees a mug. Alerting needs the
    weapon itself verified, not just the Analyst's description."""
    tools = _toolbox()
    claims = []
    tools.analyst = SimpleNamespace(describe=lambda img, q: {"description": "A white ceramic mug. No text."})
    tools.verifier = SimpleNamespace(check=lambda img, c: claims.append(c) or {
        "supported": "mug" in c, "confidence": 0.9, "reason": "stub"})
    det = _knife_det()
    tools.begin_incident(det, "x")
    out = RuleDirector(tools).investigate(det)
    assert claims == ["a knife"] and out["action"] == "log"


@pytest.mark.parametrize(("said", "claim", "same"), [
    # SmolVLM-500M's answers on COCO crops whose weapon the Verifier confirmed
    ("Cake.", "a knife", False), ("Plate with cake.", "a knife", False), ("Broccoli.", "a knife", False),
    ("Tape.", "a pair of scissors", False), ("Baseball cleats.", "a baseball bat", False),
    ("Man.", "a baseball bat", False),
    ("Knife.", "a knife", True), ("Knife and spoon.", "a knife", True), ("Knives.", "a knife", True),
    ("Scissors.", "a pair of scissors", True), ("Baseball bats.", "a baseball bat", True),
    ("A kitchen knife with a bright steel blade and a dark handle. No readable text.", "a knife", True),
    # the fine-tuned Scout's catch-all class, and an LLM Director's free-text claims
    ("Rifle.", "a weapon", True), ("Plate.", "a weapon", False),
    ("A long blade.", "a kitchen knife on the counter", True),
    ("A blue can reading ACME COLA.", "a cola can", True), ("A red mug.", "a cola can", False),
    ("", "a knife", False),
])
def test_names_claim(said, claim, same):
    assert names_claim(said, claim) is same


def _said(text: str, supported: bool, tmp_path) -> dict:
    """One incident through the orchestrator with a stubbed Analyst and Verifier."""
    s = _frozen_sherlock(tmp_path)
    s.tools.analyst = SimpleNamespace(describe=lambda img, q: {"description": text})
    s.tools.verifier = SimpleNamespace(check=lambda img, c: {"supported": supported, "reason": "stub"})
    q = BUS.subscribe()
    s._investigate(_knife_det())
    report = next(e.data for e in drain(q) if e.stage == "system" and e.msg.startswith("REPORT"))
    r = s.log[-1]
    assert (report["description"], report.get("analyst_mismatch")) == (r["description"], r.get("analyst_mismatch"))
    return r


def test_disagreement_preserves_both_results_and_routes_to_review(tmp_path):
    """Conflicting assessments stay visible and never raise an alert."""
    r = _said("Cake.", True, tmp_path)
    assert r["action"] == "log" and r["verified"] is False
    assert r["description"] == "Cake." and r["comparison"]["status"] == "disagreed"
    r = _said("Knife.", True, tmp_path)
    assert r["description"] == "Knife." and "analyst_mismatch" not in r
    r = _said("A white ceramic mug. No text.", False, tmp_path)  # dissent that agrees with the verdict
    assert r["action"] == "log" and r["description"] == "A white ceramic mug. No text."
    assert "analyst_mismatch" not in r


def test_evidence_write_failures_keep_the_report_and_the_loop(tmp_path, monkeypatch):
    blocker = tmp_path / "evidence"
    blocker.write_text("a file where the evidence folder should be")
    s = _frozen_sherlock(blocker)
    s._investigate(_knife_det())
    r = s.log[-1]
    assert r["action"] == "alert" and r["evidence"] == []
    assert r["custody_saved"] is False and r["notification"]["status"] == "failed"
    assert r["evidence_unsaved"] == ["original_frame", "face_crop", "weapon_crop", "model_context"] and "evidence_camera" not in r
    monkeypatch.setattr("sherlock.orchestrator.cv2.imwrite", lambda *a, **k: False)
    s.evidence_dir = tmp_path / "ok"
    s._investigate(_knife_det())
    assert len(s.log) == 2, "the next incident still runs"
    assert s.log[-1]["evidence"] == [] and s.log[-1]["evidence_unsaved"] == ["original_frame", "face_crop", "weapon_crop", "model_context"]


def test_finetuned_scout_classes_route():
    """YOLO_MODEL swap to the fine-tuned Scout: person/weapon/badge/bottle."""
    badge = (300, 400, 340, 430)
    scout = Scout("scene", "", dwell=0.0,
                  truth=lambda cam: [("person", PERSON), ("weapon", KNIFE), ("badge", badge), ("bottle", CUP)])
    frame = np.zeros((720, 1280, 3), np.uint8)
    scout.step(frame, "hero")
    fresh = {d.label: d for d in scout.step(frame, "hero")}
    assert fresh["weapon"].threat == "weapon" and fresh["weapon"].head == head_region(PERSON)
    assert fresh["badge"].threat == fresh["bottle"].threat == "benign"
