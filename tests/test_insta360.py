"""Insta360 Link 2 adapter against fake transports: geometry, wire format,
clamping, settle, auto-home, and failure -> CameraError -> next camera."""
import json
import math
import struct
import sys
import threading
import time
import types

import numpy as np
import pytest
from sherlock.agents.director import RuleDirector
from sherlock.agents.scout import Detection
from sherlock.agents.vision import MockAnalyst
from sherlock.camera import (
    CameraError,
    FakePTZCamera,
    Insta360Camera,
    Pose,
    build_camera,
)
from sherlock.camera.insta360 import (
    CT_PANTILT_ABSOLUTE,
    CT_ZOOM_ABSOLUTE,
    GET_MAX,
    GET_MIN,
    SET_CUR,
    FakeTransport,
    MacUSBTransport,
    PTZRanges,
    TransportError,
    Tuning,
    aim_at,
    parse_v4l2_ctrls,
)
from sherlock.config import Config
from sherlock.events import BUS
from sherlock.tools import ToolBox

W, H = 1280, 720
HALF_HFOV = 33.5  # Insta360 spec: 67 deg HFOV


class Frames:
    """Stand-in for _FrameSource; records the freshness each read asked for."""

    def __init__(self):
        self.asked: list[float] = []

    def read(self, newer_than: float = 0.0) -> np.ndarray:
        self.asked.append(newer_than)
        return np.zeros((H, W, 3), np.uint8)


def make(transport=None, **tuning) -> Insta360Camera:
    t = Tuning(**{"home_after_s": 0.0, "settle_s": 0.0, "cmd_timeout_s": 1.0, **tuning})
    return Insta360Camera("hero", "0", transport=transport or FakeTransport(), tuning=t,
                          frames=Frames())


def box_at(cx: float, cy: float, size: int = 40) -> tuple[int, int, int, int]:
    return (int(cx - size / 2), int(cy - size / 2), int(cx + size / 2), int(cy + size / 2))


# --- geometry ----------------------------------------------------------------------

def test_centered_box_keeps_aim_and_zooms_to_fill():
    cam = make(max_zoom=4.0, fill=0.6)
    pose = cam.look_at((W // 2 - 320, H // 2 - 180, W // 2 + 320, H // 2 + 180))  # half the frame
    assert pose.pan == pytest.approx(0, abs=1e-6)
    assert pose.tilt == pytest.approx(0, abs=1e-6)
    assert pose.zoom == pytest.approx(1.2)  # 0.6 fill * 2x smaller than the frame


def test_box_at_right_edge_pans_half_the_hfov():
    cam = make()
    pose = cam.look_at(box_at(W, H / 2))
    assert pose.pan == pytest.approx(HALF_HFOV, abs=0.05)
    assert pose.tilt == pytest.approx(0, abs=1e-6)


def test_box_at_top_tilts_up_by_half_the_vfov():
    cam = make()
    pose = cam.look_at(box_at(W / 2, 0))
    half_vfov = math.degrees(math.atan(math.tan(math.radians(HALF_HFOV)) * H / W))
    assert pose.tilt == pytest.approx(half_vfov, abs=0.05)
    assert pose.pan == pytest.approx(0, abs=1e-6)


def test_zoomed_view_maps_offsets_through_the_narrower_fov():
    cam = make()
    cam.move(pan=20, tilt=0, zoom=2)
    cam.wide_frame()
    pose = cam.look_at(box_at(W, H / 2))
    expected = 20 + math.degrees(math.atan(math.tan(math.radians(HALF_HFOV)) / 2))
    assert pose.pan == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("pan,tilt,zoom", [(0, 0, 1), (35, 20, 2), (-80, -40, 3.5), (120, 25, 1)])
def test_centered_box_is_a_fixed_point_at_any_pose(pan, tilt, zoom):
    got = aim_at(box_at(W / 2, H / 2), (W, H), Pose(pan, tilt, zoom), 67.0)
    assert got == pytest.approx((pan, tilt), abs=1e-6)


def test_aims_from_the_pose_the_scout_frame_was_taken_at():
    cam = make()
    cam.wide_frame()            # Scout grabs a frame at home
    cam.move(pan=50, tilt=10)   # gimbal moves before the Director acts
    pose = cam.look_at(box_at(W / 2, H / 2))
    assert (pose.pan, pose.tilt) == pytest.approx((0, 0), abs=1e-6)


# --- wire format and limits -------------------------------------------------------

def test_degrees_become_arcseconds_with_configurable_signs():
    tr = FakeTransport()
    cam = make(tr, pan_sign=-1.0)
    cam.move(pan=10, tilt=-5, zoom=2)
    assert ("pantilt", -36000, -18000) in tr.calls
    assert tr.calls[-1] == ("zoom", 200)


def test_clamps_to_ranges_the_camera_reports():
    tr = FakeTransport(fake_ranges=PTZRanges(pan=(-90 * 3600, 90 * 3600),
                                             tilt=(-30 * 3600, 45 * 3600), zoom=(100, 400)))
    cam = make(tr, max_zoom=3.0)
    pose = cam.move(pan=150, tilt=-80, zoom=9)
    assert (pose.pan, pose.tilt, pose.zoom) == (90, -30, 3.0)
    assert tr.calls[-1] == ("zoom", 300)


def test_mac_usb_transport_wire_format(monkeypatch):
    sent: list[tuple] = []
    replies = {(GET_MIN, CT_PANTILT_ABSOLUTE): struct.pack("<ii", -522000, -324000),
               (GET_MAX, CT_PANTILT_ABSOLUTE): struct.pack("<ii", 522000, 108000),
               (GET_MIN, CT_ZOOM_ABSOLUTE): struct.pack("<H", 100),
               (GET_MAX, CT_ZOOM_ABSOLUTE): struct.pack("<H", 400)}
    opened: list[int] = []

    def open_vc_interface(vid, pid, iface_num):
        opened.append(pid)
        return types.SimpleNamespace(close=lambda: None) if pid == 0x4C04 else None

    def control_request(handle, bm, req, w_value, w_index, data=b"", read_length=0):
        sent.append((bm, req, w_value, w_index, bytes(data)))
        return True, replies.get((req, w_value >> 8), b"")

    fake = types.ModuleType("link_usb_macos")
    fake.open_vc_interface, fake.control_request = open_vc_interface, control_request
    monkeypatch.setitem(sys.modules, "link_usb_macos", fake)

    tr = MacUSBTransport()
    tr.open()
    assert opened[0] == 0x4C04 and tr.pid == 0x4C04  # Link 2 first, not link-ctl's 0x4c01 default
    assert tr.ranges() == PTZRanges(pan=(-522000, 522000), tilt=(-324000, 108000), zoom=(100, 400))
    sent.clear()
    tr.set_pantilt(36000, -7200)
    tr.set_zoom(250)
    assert sent == [(0x21, SET_CUR, 0x0D00, 0x0100, struct.pack("<ii", 36000, -7200)),
                    (0x21, SET_CUR, 0x0B00, 0x0100, struct.pack("<H", 250))]


def test_v4l2_ctrl_listing_parses_ranges():
    text = """
                   pan_absolute 0x009a0908 (int)    : min=-522000 max=522000 step=3600 default=0 value=0
                  tilt_absolute 0x009a0909 (int)    : min=-324000 max=324000 step=3600 default=0 value=0
                  zoom_absolute 0x009a090d (int)    : min=100 max=400 step=1 default=100 value=100
"""
    c = parse_v4l2_ctrls(text)
    assert c["pan_absolute"]["min"] == -522000 and c["zoom_absolute"]["max"] == 400


# --- timing ------------------------------------------------------------------------

def test_snapshot_waits_for_the_gimbal_to_settle():
    cam = make(settle_s=0.2, slew_deg_s=1e9)
    cam.move(pan=10)
    t0 = time.monotonic()
    cam.snapshot()
    assert time.monotonic() - t0 >= 0.18
    assert cam.src.asked[-1] == pytest.approx(cam._settle_until)  # asks for a post-settle frame


class RetainedFrames:
    """Like _FrameSource: keeps the last frame and its capture time, and only
    captures a new one when a caller asks for something newer. Each frame's
    pixels hold the pan the gimbal was last sent, so a test can tell which pose
    a frame really shows."""

    def __init__(self, transport: FakeTransport):
        self.tr = transport
        self.frame, self.stamp = self._capture(), time.monotonic()
        self.on_read = None  # hook to fire once, e.g. an auto-home mid-read

    def _capture(self) -> np.ndarray:
        pans = [c[1] for c in self.tr.calls if c[0] == "pantilt"]
        return np.full((H, W, 3), 100 + (pans[-1] if pans else 0), np.uint8)

    def read(self, newer_than: float = 0.0) -> np.ndarray:
        if self.on_read:
            hook, self.on_read = self.on_read, None
            hook()
        if self.stamp < newer_than:
            time.sleep(max(0.0, newer_than - time.monotonic()))
            self.frame, self.stamp = self._capture(), time.monotonic()
        return self.frame


def test_scout_frames_are_captured_after_settling_at_the_pose_they_carry():
    tr = FakeTransport()
    frames = RetainedFrames(tr)
    t = Tuning(home_after_s=0.0, settle_s=0.1, cmd_timeout_s=1.0, units_per_deg=1)
    cam = Insta360Camera("hero", "0", transport=tr, tuning=t, frames=frames)
    frames.frame, frames.stamp = frames._capture(), time.monotonic()  # home view, cached
    cam.move(pan=30)
    f = cam.wide_frame()  # the cached home frame must not come back labelled pan 30
    assert int(f[0, 0, 0]) == 130 and cam._view_pose.pan == 30
    # auto-home fires while the Scout is reading: the frame read across the move
    # is dropped and the next one is labelled with the home pose
    def home_mid_read() -> None:  # the gimbal moves and the stream delivers it
        cam._go_home(cam._home_gen)
        frames.frame, frames.stamp = frames._capture(), time.monotonic()

    frames.on_read = home_mid_read
    f = cam.wide_frame()
    assert int(f[0, 0, 0]) == 100 and cam._view_pose.pan == 0


def test_returns_home_when_idle_and_new_moves_reset_the_clock():
    tr = FakeTransport()
    cam = make(tr, home_after_s=0.3)
    cam.move(pan=30, zoom=2)
    time.sleep(0.2)
    cam.move(pan=40)  # re-arms the timer
    time.sleep(0.2)
    assert cam.pose.pan == 40
    time.sleep(0.3)
    assert cam.pose == Pose(0, 0, 1)
    assert tr.calls[-2:] == [("pantilt", 0, 0), ("zoom", 100)]


# --- failure paths -----------------------------------------------------------------

def test_dead_gimbal_raises_camera_error_but_construction_survives():
    cam = make(FakeTransport(fail=True))
    assert cam.describe()["ptz"].startswith("offline")
    with pytest.raises(CameraError, match="gimbal not responding"):
        cam.move(pan=10)
    with pytest.raises(CameraError):
        cam.look_at(box_at(W, H / 2))


def test_silent_gimbal_times_out_as_camera_error():
    cam = make(FakeTransport(hang_s=0.5), cmd_timeout_s=0.1)
    t0 = time.monotonic()
    with pytest.raises(CameraError, match="no answer"):
        cam.move(pan=10)
    assert time.monotonic() - t0 < 0.4


class GatedTransport(FakeTransport):
    """SETs block on a gate; logs open/close/SET start and end in order."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        self.gate.set()
        self.log: list[str] = []

    def open(self) -> None:
        self.log.append("open")

    def close(self) -> None:
        self.log.append("close")

    def set_pantilt(self, pan: int, tilt: int) -> None:
        self.log.append(f"set {pan} start")
        self.gate.wait()
        self.log.append(f"set {pan} end")


def test_timed_out_set_quarantines_the_transport_until_it_returns():
    tr = GatedTransport()
    cam = make(tr, cmd_timeout_s=0.1, units_per_deg=1)
    tr.gate.clear()  # the next SET hangs past the timeout
    with pytest.raises(CameraError, match="no answer within"):
        cam.move(pan=10)
    with pytest.raises(CameraError, match="still has no answer"):
        cam.move(pan=20)  # retry while the hung SET is still on the wire
    assert "close" not in tr.log[tr.log.index("set 10 start"):]
    tr.gate.set()
    deadline = time.monotonic() + 2
    while "set 10 end" not in tr.log and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cam.move(pan=20).pan == 20
    tail = tr.log[tr.log.index("set 10 start"):]
    assert tail == ["set 10 start", "set 10 end", "close", "open", "set 20 start", "set 20 end"]


class ZoomFailsTransport(FakeTransport):
    """Pan/tilt SETs land; zoom SETs fail while `zoom_fails` is set."""

    zoom_fails = False

    def set_zoom(self, zoom: int) -> None:
        if self.zoom_fails:
            raise TransportError("simulated: zoom rejected")
        super().set_zoom(zoom)


def test_partial_failure_resends_the_whole_pose_on_the_next_move():
    tr = ZoomFailsTransport()
    cam = make(tr, units_per_deg=1)
    tr.zoom_fails = True
    with pytest.raises(CameraError):
        cam.move(pan=30, zoom=2)  # pan reached the gimbal, zoom did not
    assert tr.calls[-1] == ("pantilt", 30, 0)
    tr.zoom_fails = False
    tr.calls.clear()
    assert cam.move(pan=0, zoom=1) == cam.home
    assert tr.calls == [("pantilt", 0, 0), ("zoom", 100)]  # not skipped as "no change"


def test_reconnects_when_the_gimbal_comes_back():
    tr = FakeTransport(fail=True)
    cam = make(tr)
    with pytest.raises(CameraError):
        cam.move(pan=10)
    tr.fail = False
    assert cam.move(pan=10).pan == 10
    assert cam.describe()["ptz"] == "fake"


def test_covered_lens_is_loud():
    cam = make()
    cam.covered = True
    with pytest.raises(CameraError, match="lens covered"):
        cam.snapshot()


def test_director_falls_back_to_next_camera_when_gimbal_dies():
    hero = make(FakeTransport(fail=True))
    tools = ToolBox({"hero": hero, "scout1": FakePTZCamera("scout1", None)}, MockAnalyst())
    r = tools.call("look_at", {"camera": "hero", "bbox": [600, 300, 700, 400]})
    assert "gimbal not responding" in r["error"]
    before = BUS.stats.get("recoveries", 0)
    report = RuleDirector(tools).investigate(Detection("bottle", 0.9, (600, 300, 700, 400), "hero"))
    assert report["done"] and report["claim"] != "all cameras failed"
    assert BUS.stats["recoveries"] > before


# --- registration ------------------------------------------------------------------

def test_backend_name_registers_and_parses(monkeypatch):
    monkeypatch.setenv("CAMERAS", "hero:insta360:0,scout1:fake:0")
    specs = Config().cameras
    assert [(c.name, c.backend, c.source) for c in specs] == [("hero", "insta360", "0"),
                                                             ("scout1", "fake", "0")]
    cam = build_camera("hero", "insta360", "synthetic")  # --synthetic runs need no hardware
    assert isinstance(cam, Insta360Camera) and isinstance(cam.transport, FakeTransport)
    cam.close()


def test_synthetic_runs_simulate_the_optics_so_look_at_lands_on_the_box():
    class Marker(Frames):
        def read(self, newer_than: float = 0.0) -> np.ndarray:
            f = np.zeros((H, W, 3), np.uint8)
            f[500:560, 1000:1060] = 255  # off-centre target
            return f

    cam = Insta360Camera("hero", "0", transport=FakeTransport(), frames=Marker(),
                         tuning=Tuning(home_after_s=0, settle_s=0), simulate_optics=True)
    cam.wide_frame()
    cam.look_at((1000, 500, 1060, 560))
    img = cam.snapshot()
    assert img[H // 2, W // 2].min() == 255  # target is dead centre
    assert (img == 255).mean() > 0.04        # and magnified 4x


# --- name-based camera selection ---------------------------------------------------

def test_name_source_resolves_against_opencv_order():
    from sherlock.camera.base import resolve_source

    devices = [(0, "MacBook Pro Camera"), (1, "OBS Virtual Camera"), (2, "Insta360 Link 2")]
    assert resolve_source("name:Insta360 Link 2", devices) == "2"
    assert resolve_source("name:macbook pro camera", devices) == "0"   # case-insensitive
    assert resolve_source("name:OBS", devices) == "1"                  # unique substring
    assert resolve_source("1", devices) == "1"                         # indexes still work
    with pytest.raises(CameraError, match="no camera"):
        resolve_source("name:Link 2C", devices)
    with pytest.raises(CameraError, match="several cameras"):
        resolve_source("name:Camera", devices)


def test_macos_order_is_uniqueid_sorted_like_opencv(monkeypatch):
    from sherlock.camera import base

    listing = {"SPCameraDataType": [
        {"_name": "OBS Virtual Camera", "spcamera_unique-id": "7626645E-4425"},
        {"_name": "MacBook Pro Camera", "spcamera_unique-id": "6C707041-05AC"}]}
    monkeypatch.setattr(base.sys, "platform", "darwin")
    monkeypatch.setattr(base.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps(listing), stderr=""))
    assert base.list_video_devices() == [(0, "MacBook Pro Camera"), (1, "OBS Virtual Camera")]


def test_linux_keeps_real_video_numbers_across_gaps(monkeypatch, tmp_path):
    from sherlock.camera import base

    for n, name in ((2, "Insta360 Link 2"), (5, "Integrated Camera"), (10, "Insta360 Link 2 meta")):
        (tmp_path / f"video{n}").mkdir()
        (tmp_path / f"video{n}" / "name").write_text(name + "\n")
    monkeypatch.setattr(base.sys, "platform", "linux")
    monkeypatch.setattr(base, "V4L2_SYSFS", tmp_path)
    devices = base.list_video_devices()
    assert devices == [(2, "Insta360 Link 2"), (5, "Integrated Camera"), (10, "Insta360 Link 2 meta")]
    assert base.resolve_source("name:Insta360 Link 2", devices) == "2"  # /dev/video2, not 0
    assert base.resolve_source("name:integrated", devices) == "5"


def test_cameras_env_accepts_name_sources(monkeypatch):
    monkeypatch.setenv("CAMERAS", "hero:insta360:name:Insta360 Link 2,scout1:fake:name:MacBook Pro Camera")
    assert [(c.name, c.backend, c.source) for c in Config().cameras] == [
        ("hero", "insta360", "name:Insta360 Link 2"),
        ("scout1", "fake", "name:MacBook Pro Camera")]
