"""SCOUT_MISLABEL: a Scout false positive on demand, for demos and tests. Off unless
set; when on, the raw class is renamed before the Scout classifies it, and every
injected detection is announced so nobody takes it for a real one."""
import queue

import numpy as np
import pytest
from sherlock.agents.scout import Scout, head_region, parse_mislabel
from sherlock.events import BUS

PERSON = (300, 100, 500, 700)
CUP = (520, 300, 600, 400)
FRAME = np.zeros((720, 1280, 3), np.uint8)


def scene_scout(**kw) -> Scout:
    return Scout("scene", "", dwell=0.0, truth=lambda cam: [("person", PERSON), ("cup", CUP)], **kw)


def first_event(scout: Scout) -> dict:
    scout.step(FRAME, "hero")  # first sighting starts the track
    (det,) = scout.step(FRAME, "hero")
    return det


def drain(q: queue.Queue) -> list[str]:
    out = []
    while not q.empty():
        ev = q.get_nowait()
        out.append(f"{ev.stage}: {ev.msg}")
    return out


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SCOUT_MISLABEL", raising=False)
    q = BUS.subscribe()
    scout = scene_scout()
    det = first_event(scout)
    assert scout.mislabel == {}
    assert (det.label, det.threat, det.mislabeled_from) == ("cup", "benign", None)
    assert not [m for m in drain(q) if "mislabel" in m]


def test_renames_the_raw_class_before_the_scout_classifies_it(monkeypatch):
    monkeypatch.setenv("SCOUT_MISLABEL", "cup=knife")
    det = first_event(scene_scout())
    assert (det.label, det.threat, det.mislabeled_from) == ("knife", "weapon", "cup")
    # Treated exactly like a real detector mistake: aimed at the nearest person's head.
    assert det.person == PERSON and det.head == head_region(PERSON)


def test_announced_at_startup_and_on_every_injected_detection(monkeypatch):
    monkeypatch.setenv("SCOUT_MISLABEL", "cup=knife")
    q = BUS.subscribe()
    scout = scene_scout()
    assert drain(q) == ["system: Scout mislabel injected: cup -> knife (test fault, SCOUT_MISLABEL)"]
    first_event(scout)
    rows = drain(q)
    injected = "system: Scout mislabel injected: cup -> knife on hero (test fault, not a real detection)"
    assert rows[0] == injected, "the warning comes before the detection it explains"
    assert rows[1].startswith("scout: new knife on hero")
    assert scout.step(FRAME, "hero") == [] and not drain(q), "once per object, not once per frame"


def test_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("SCOUT_MISLABEL", "cup=knife")
    assert first_event(scene_scout(mislabel="")).label == "cup"


@pytest.mark.parametrize("spec, expected", [
    ("", {}),
    ("cup=knife", {"cup": "knife"}),
    (" cup = knife , bottle=scissors ,", {"cup": "knife", "bottle": "scissors"}),
])
def test_parse(spec, expected):
    assert parse_mislabel(spec) == expected


@pytest.mark.parametrize("spec", ["cup", "=knife", "cup=", "cup=knife,bottle"])
def test_parse_rejects_malformed_entries(spec):
    with pytest.raises(ValueError, match="SRC=DST"):
        parse_mislabel(spec)
