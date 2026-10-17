"""The dashboard gets the model's object input before inference, never face evidence."""
import base64
from types import SimpleNamespace

import cv2
import numpy as np
from sherlock.agents.scout import Detection
from sherlock.camera import FakePTZCamera
from sherlock.tools import ToolBox, _preview


def decode(event):
    raw = base64.b64decode(event["image"].split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_previews_precede_calls_and_verifier_keeps_object_after_face_snapshot(monkeypatch):
    events = []
    monkeypatch.setattr("sherlock.tools.BUS.emit", lambda stage, msg, **data: events.append((stage, data)))
    object_image = np.full((60, 40, 3), (0, 0, 255), np.uint8)
    face_image = np.full((60, 40, 3), (255, 0, 0), np.uint8)
    cam = FakePTZCamera("hero", None)
    monkeypatch.setattr(cam, "snapshot", lambda: object_image.copy())
    monkeypatch.setattr(cam, "view_box", lambda box, size: (0, 0, size[0], size[1]))

    def observe(role, image):
        stage, event = events[-1]
        assert stage == "model_input" and event["role"] == role
        np.testing.assert_array_equal(decode(event), image)

    def describe(image, question):
        observe("analyst", image)
        return {"description": "knife"}

    def verify(image, claim):
        observe("verifier", image)
        np.testing.assert_array_equal(image, object_image)
        return {"supported": True}

    tools = ToolBox({"hero": cam}, SimpleNamespace(describe=describe), verifier=SimpleNamespace(check=verify))
    tools.begin_incident(Detection("knife", 0.9, (0, 0, 40, 60), "hero", threat="weapon"), "preview")
    tools.aim["hero"] = (0, 0, 40, 60)
    tools.t_snapshot("hero", "weapon")
    before_face = len(events)
    monkeypatch.setattr(cam, "snapshot", lambda: face_image.copy())
    tools.t_snapshot("hero", "face")
    assert len(events) == before_face, "face evidence must not emit a model preview"
    tools.check_claim("a knife")
    assert [event[1]["role"] for event in events] == ["analyst", "verifier"]


def test_preview_is_bounded_without_changing_the_model_input(monkeypatch):
    events = []
    monkeypatch.setattr("sherlock.tools.BUS.emit", lambda stage, msg, **data: events.append(data))
    image = np.full((720, 1280, 3), (30, 60, 90), np.uint8)
    original = image.copy()
    _preview("analyst", image)
    assert decode(events[0]).shape == (144, 256, 3)
    np.testing.assert_array_equal(original, image)
