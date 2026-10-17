from types import SimpleNamespace

from sherlock.agents import scout
from sherlock.agents.scout import Detection, _Tracker
from test_insta360 import make
from test_weapon_flow import _frozen_sherlock


def test_knife_cannot_inherit_an_already_reported_remote_track(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(scout.time, "time", lambda: clock[0])
    tracker = _Tracker()
    box = (10, 10, 50, 100)
    remote = Detection("remote", 0.8, box, "hero")
    knife = Detection("knife", 0.8, box, "hero")
    assert tracker.update([remote]) == []
    clock[0] = 0.7
    assert tracker.update([remote]) == [remote]
    assert tracker.update([knife]) == []
    clock[0] = 0.85
    assert tracker.update([knife]) == [knife]
    assert tracker.update([knife]) == []


def test_returning_knife_gets_a_new_track_after_expiry(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(scout.time, "time", lambda: clock[0])
    tracker = _Tracker(dwell=0)
    knife = Detection("knife", 0.8, (10, 10, 50, 100), "hero")
    tracker.update([knife])
    assert tracker.update([knife]) == [knife]
    clock[0] = 4
    assert tracker.update([knife]) == []
    assert tracker.update([knife]) == [knife]


def test_benign_objects_do_not_wait_for_the_cloud_director(tmp_path):
    s = _frozen_sherlock(tmp_path)

    def unexpected(_):
        raise AssertionError("benign detection reached the cloud Director")

    s.director = SimpleNamespace(investigate=unexpected)
    s._investigate(Detection("cup", 0.8, (10, 10, 50, 100), "hero", threat="benign"))
    assert s.log[-1]["action"] == "ignore"
    assert "error" not in s.log[-1]


def test_live_preview_does_not_wait_for_evidence_settling(monkeypatch):
    cam = make()
    try:
        cam._settle_until = float("inf")
        view_pose = cam._view_pose

        def unexpected(_):
            raise AssertionError("preview waited for the gimbal")

        monkeypatch.setattr("sherlock.camera.insta360.time.sleep", unexpected)
        assert cam.preview_frame().shape == (720, 1280, 3)
        assert cam.src.asked[-1] == 0
        assert cam._view_pose == view_pose
    finally:
        cam.close()
