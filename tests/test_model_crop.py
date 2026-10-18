"""The crop a model sees is the aimed box plus a margin, never the whole zoomed view.

On 2026-10-17 the zoomed view around a cup (COCO val2017 000000199771) held a real knife's
handle, and the Verifier confirmed "a knife"; around a wide, flat knife (000000394940) it
held a child's mouth and chin. Each scene below puts the intruder inside the old view and
outside the box plus margin, and checks the full view still shows it."""
from types import SimpleNamespace

import numpy as np
from sherlock.agents.scout import Detection, head_region
from sherlock.camera import FakePTZCamera
from sherlock.camera.insta360 import FakeTransport, Insta360Camera, Tuning
from sherlock.tools import MODEL_CROP_FLOOR, MODEL_CROP_MARGIN, ToolBox, _crop

W, H = 1280, 720
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)
CUP = (600, 300, 690, 440)        # the mislabelled box: 90 x 140, like the 199771 cup
NEIGHBOUR = (720, 320, 800, 420)  # 30 px to its right: past 20% of 90, inside the old view


class Still:
    """A fixed frame for the software and the simulated Insta360 optics."""

    def __init__(self, paint: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]]):
        self.frame = np.zeros((H, W, 3), np.uint8)
        for (x1, y1, x2, y2), colour in paint:
            self.frame[y1:y2, x1:x2] = colour

    def read(self, newer_than: float = 0.0) -> np.ndarray:
        return self.frame.copy()


def has(img: np.ndarray, colour) -> bool:
    return bool((img == colour).all(axis=2).any())


def weapon_crop(cam, det: Detection) -> tuple[ToolBox, np.ndarray]:
    seen = []
    tools = ToolBox({"hero": cam}, SimpleNamespace(describe=lambda img, q: seen.append(img) or {"description": "x"}))
    tools.begin_incident(det, "x")
    assert tools.call("look_at", {"camera": "hero", "bbox": list(det.bbox)})["ok"]
    assert "error" not in tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert seen[0] is tools.evidence["weapon_crop"], "the Analyst sees exactly the stored crop"
    return tools, seen[0]


def assert_whole_box_and_no(crop: np.ndarray, intruder) -> None:
    assert has(crop, RED), "the weapon is in the crop"
    edges = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    assert not (edges == RED).all(axis=1).any(), "margin on every side: the box is not cut off"
    assert not has(crop, intruder)


def test_neighbour_past_the_margin_is_excluded_on_the_software_ptz():
    cam = FakePTZCamera("hero", None)
    cam.src = Still([(CUP, RED), (NEIGHBOUR, GREEN)])
    _, crop = weapon_crop(cam, Detection("knife", 0.9, CUP, "hero", threat="weapon"))
    assert has(cam.snapshot(), GREEN), "the old model crop, the whole zoomed view, held the neighbour"
    assert_whole_box_and_no(crop, GREEN)


def test_neighbour_past_the_margin_is_excluded_on_the_insta360_path():
    # Simulated optics invert aim_at exactly, so the crop comes from view_box's projection:
    # the box centred and magnified by the zoom, as on the hardware.
    cam = Insta360Camera("hero", "0", transport=FakeTransport(), frames=Still([(CUP, RED), (NEIGHBOUR, GREEN)]),
                         tuning=Tuning(home_after_s=0, settle_s=0), simulate_optics=True)
    _, crop = weapon_crop(cam, Detection("knife", 0.9, CUP, "hero", threat="weapon"))
    assert has(cam.snapshot(), GREEN)
    assert_whole_box_and_no(crop, GREEN)


def test_face_above_a_raised_wide_knife_stays_out_of_the_model_crop():
    person = (560, 100, 760, 700)
    head = head_region(person)  # (606, 100, 714, 220)
    knife = (560, 260, 760, 290)  # held flat under the chin: wide, so the zoom view runs tall
    cam = FakePTZCamera("hero", None)
    cam.src = Still([(head, BLUE), (knife, RED)])
    tools, crop = weapon_crop(cam, Detection("knife", 0.9, knife, "hero", threat="weapon", person=person, head=head))
    assert has(cam.snapshot(), BLUE), "the old model crop held the face"
    assert_whole_box_and_no(crop, BLUE)
    # The aim rules are unchanged: a head zoom still never reaches a model.
    tools.call("look_at", {"camera": "hero", "bbox": list(head)})
    assert "error" in tools.call("snapshot", {"camera": "hero", "subject": "weapon"})
    assert tools.call("snapshot", {"camera": "hero", "subject": "face"})["analyzed"] is False


def test_thin_box_keeps_context_on_its_narrow_sides():
    upright_knife = (500, 400, 524, 550)  # 24 x 150
    crop = _crop(np.zeros((1000, 1000, 3), np.uint8), upright_knife, MODEL_CROP_MARGIN, MODEL_CROP_FLOOR)
    # Sides: 10% of the long side (15 px) beats 20% of 24; ends: 20% of 150 (30 px).
    assert crop.shape[:2] == (150 + 2 * 30, 24 + 2 * 15)
