"""Every crop bound for a model has its faces filled before it leaves the toolbox; the face
crop, evidence for a human only, is stored as captured."""
import hashlib
import queue
from types import SimpleNamespace

import cv2
import numpy as np
from sherlock.agents.scout import Detection
from sherlock.camera import FakePTZCamera
from sherlock.events import BUS
from sherlock.privacy import FILL, YUNET, FaceMasker, fill
from sherlock.tools import MODEL_CROP_FLOOR, MODEL_CROP_MARGIN, ToolBox, _crop

W, H = 1280, 720
RED, BLUE = (0, 0, 255), (255, 0, 0)
KNIFE = (500, 400, 700, 440)
FACE = (520, 380, 560, 400)  # inside the knife box's 20% margin: a mouth just above the blade
HEAD = (560, 100, 680, 220)


class Still:
    def __init__(self, paint):
        self.frame = np.zeros((H, W, 3), np.uint8)
        for (x1, y1, x2, y2), colour in paint:
            self.frame[y1:y2, x1:x2] = colour

    def read(self, newer_than: float = 0.0) -> np.ndarray:
        return self.frame.copy()


class BlueFaces(FaceMasker):
    """Takes blue for a face, so a test knows exactly where the faces are in any crop."""

    def __init__(self):  # no network to load
        self.seen: list[np.ndarray] = []

    def faces(self, img):
        self.seen.append(img)
        ys, xs = np.nonzero((img == BLUE).all(axis=2))
        return [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)] if len(xs) else []


def has(img: np.ndarray, colour) -> bool:
    return bool((img == colour).all(axis=2).any())


def toolbox(masker=None, paint=((KNIFE, RED), (FACE, BLUE), (HEAD, BLUE))):
    cam = FakePTZCamera("hero", None)
    cam.src = Still(list(paint))
    shown = {"analyst": [], "verifier": []}
    analyst = SimpleNamespace(describe=lambda img, q: shown["analyst"].append(img) or {"description": "A knife."})
    verifier = SimpleNamespace(check=lambda img, claim: shown["verifier"].append(img) or {"supported": True})
    tools = ToolBox({"hero": cam}, analyst, verifier=verifier, masker=masker)
    tools.begin_incident(Detection("knife", 0.9, KNIFE, "hero", threat="weapon", head=HEAD), "mask")
    return tools, shown


def snapshot_event(q: queue.Queue):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return next(e for e in events if e.stage == "tool" and e.data.get("tool") == "snapshot"
                and e.data.get("incident") == "mask")


def test_a_face_in_the_weapon_crop_is_filled_before_any_model_sees_it():
    masker = BlueFaces()
    tools, shown = toolbox(masker)
    q = BUS.subscribe()
    tools.call("look_at", {"camera": "hero", "bbox": list(KNIFE)})
    assert "error" not in tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert has(masker.seen[0], BLUE), "the unmasked crop held the face"
    tools.call("verify", {"claim": "a knife"})
    for img in (shown["analyst"][0], shown["verifier"][0], tools.evidence["weapon_crop"]):
        assert not has(img, BLUE) and has(img, FILL) and has(img, RED), "face gone, knife kept"
    snap = snapshot_event(q)
    assert snap.data["faces_masked"] == 1 and snap.data["mask_ms"] >= 0
    assert snap.msg.endswith(" · 1 face masked")
    assert "faces_masked" not in snap.data["result"], "the count is for the log, not the Director"


def test_a_crop_without_a_face_reaches_the_models_unchanged():
    masker = FaceMasker()  # the shipped YuNet
    tools, shown = toolbox(masker, paint=((KNIFE, RED),))
    q = BUS.subscribe()
    tools.call("look_at", {"camera": "hero", "bbox": list(KNIFE)})
    tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    crop, cam = shown["analyst"][0], tools.cameras["hero"]
    view = cam.snapshot()
    unmasked = _crop(view, cam.view_box(KNIFE, (view.shape[1], view.shape[0])), MODEL_CROP_MARGIN, MODEL_CROP_FLOOR)
    assert masker.faces(crop) == [] and np.array_equal(crop, unmasked), "the box plus margin, nothing filled"
    snap = snapshot_event(q)
    assert snap.data["faces_masked"] == 0 and "masked" not in snap.msg


def test_the_face_crop_is_stored_as_captured_and_never_masked():
    masker = BlueFaces()
    tools, shown = toolbox(masker)
    q = BUS.subscribe()
    tools.call("look_at", {"camera": "hero", "bbox": list(HEAD)})
    out = tools.call("snapshot", {"camera": "hero", "subject": "face"})
    assert out == {"evidence_id": "face_crop", "stored": True, "analyzed": False}
    assert has(tools.evidence["face_crop"], BLUE) and not has(tools.evidence["face_crop"], FILL)
    assert masker.seen == [] and shown == {"analyst": [], "verifier": []}
    snap = snapshot_event(q)
    assert "faces_masked" not in snap.data


def test_a_masking_failure_sends_nothing_to_a_model():
    class Broken(BlueFaces):
        def faces(self, img):
            raise cv2.error("detector fell over")

    tools, shown = toolbox(Broken())
    tools.call("look_at", {"camera": "hero", "bbox": list(KNIFE)})
    out = tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert "face masking failed" in out["error"]
    assert shown["analyst"] == [] and "weapon_crop" not in tools.evidence


def test_fill_grows_each_face_and_stops_at_the_crop_edge():
    img = np.zeros((100, 100, 3), np.uint8)
    out = fill(img, [(-10, 40, 30, 80)])  # a face cut by the left edge: 40 x 40, 10 px margin
    filled = (out == FILL).all(axis=2)
    ys, xs = np.nonzero(filled)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (0, 39, 30, 89)
    assert not img.any(), "the input is left alone"


def test_the_shipped_detector_is_the_pinned_yunet():
    assert hashlib.sha256(YUNET.read_bytes()).hexdigest() == (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
    assert (YUNET.parent / "LICENSE-yunet").read_text().startswith("MIT License")
    assert FaceMasker().faces(np.full((48, 300, 3), 90, np.uint8)) == []
