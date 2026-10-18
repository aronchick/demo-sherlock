from types import SimpleNamespace

import numpy as np
import pytest
from sherlock.agents.director import RuleDirector
from sherlock.agents.scout import Detection, Scout, classify
from sherlock.events import BUS
from test_weapon_flow import KNIFE, _frozen_sherlock, chain, drain


@pytest.mark.parametrize("label", ["cap", "fedora", "hat"])
@pytest.mark.parametrize("supported", [True, False])
def test_hat_gets_object_evidence_without_a_weapon_alert(tmp_path, label, supported):
    s = _frozen_sherlock(tmp_path)
    calls = []
    director = RuleDirector(s.tools)

    def inspect(detection):
        calls.append(detection.label)
        return director.investigate(detection)

    s.director = SimpleNamespace(investigate=inspect)
    s.tools.analyst = SimpleNamespace(describe=lambda image, question: {"description": f"A {label}."})
    s.tools.verifier = SimpleNamespace(check=lambda image, claim: {"supported": supported, "confidence": .9})
    detection = Detection(label, .9, KNIFE, "hero", threat=classify(label))
    q = BUS.subscribe()
    s._investigate(detection)
    report = s.log[-1]
    assert calls == [label], "hat inspection must reach the configured Director"
    assert report["trigger"]["threat"] == "inspect"
    assert report["action"] == "log"
    assert "error" not in report
    assert "object_crop" in report["evidence"]
    assert "weapon_crop" not in report["evidence"]
    assert "face_crop" not in report["evidence"]
    assert chain(drain(q), report["incident"])[:2] == ["look_at", "snapshot"]
    record = s.incidents.read(report["incident"])
    assert record["notification"]["status"] == "not_sent"
    assert record["verification"] == {"signature": "verified", "artifacts": "verified", "history": "verified"}


def test_hat_scout_detection_and_overlay_keep_weapon_routing(tmp_path):
    scout = Scout("scene", "", dwell=0, truth=lambda camera: [
        ("cap", (10, 10, 50, 50)), ("knife", (100, 100, 140, 200)),
        ("cup", (200, 200, 240, 260))])
    frame = np.zeros((720, 1280, 3), np.uint8)
    scout.step(frame, "hero")
    detections = scout.step(frame, "hero")
    assert {d.label: d.threat for d in detections} == {
        "cap": "inspect", "knife": "weapon", "cup": "benign"}
    s = _frozen_sherlock(tmp_path)
    s.scout.last_dets["hero"] = detections
    assert s._annotate("hero", frame).any()
