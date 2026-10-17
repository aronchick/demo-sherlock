"""The OBSBOT adapter against a fake IOKit module and a fake gimbal. None of this has run
on a real OBSBOT yet; scripts/obsbot_probe.py is the hardware check."""
import struct
import sys
import types

import pytest
from sherlock.camera import ObsbotCamera, build_camera
from sherlock.camera.base import CameraError, resolve_source
from sherlock.camera.insta360 import SET_CUR, FakeTransport, PTZRanges, Tuning
from sherlock.camera.obsbot import OBSBOT_VID, ObsbotUSBTransport, tracking_payload

FAST = {"home_after_s": 0.0, "settle_s": 0.0, "cmd_timeout_s": 1.0}


def fake_iokit(monkeypatch, present=(OBSBOT_VID, 0xFEF9), xu_ok=True):
    """link_usb_macos with one camera on the bus; records every control transfer."""
    opened, sent = [], []

    def open_vc_interface(vid, pid, iface_num):
        opened.append((vid, pid))
        return types.SimpleNamespace(close=lambda: None) if (vid, pid) == present else None

    def control_request(handle, bm, req, w_value, w_index, data=b"", read_length=0):
        sent.append((bm, req, w_value, w_index, bytes(data)))
        return (xu_ok or w_index >> 8 != 2), b"\0" * read_length

    fake = types.ModuleType("link_usb_macos")
    fake.open_vc_interface, fake.control_request = open_vc_interface, control_request
    monkeypatch.setitem(sys.modules, "link_usb_macos", fake)
    return opened, sent


def test_opens_the_obsbot_by_its_ids_and_switches_tracking_off_first(monkeypatch):
    opened, sent = fake_iokit(monkeypatch)
    tr = ObsbotUSBTransport()
    tr.open()
    assert opened == [(0x3564, 0xFEF8), (0x3564, 0xFEF9)] and tr.pid == 0xFEF9
    bm, req, w_value, w_index, payload = sent[0]
    assert (bm, req, w_value, w_index) == (0x21, SET_CUR, 0x06 << 8, 2 << 8), "XU 2, selector 6"
    assert len(payload) == 60 and payload[:4] == bytes([0x16, 2, 0, 0]) and not any(payload[4:])
    tr.set_pantilt(36000, -7200)
    tr.set_zoom(50)
    assert sent[1:] == [(0x21, SET_CUR, 0x0D00, 0x0100, struct.pack("<ii", 36000, -7200)),
                        (0x21, SET_CUR, 0x0B00, 0x0100, struct.pack("<H", 50))]


def test_tracking_is_switched_off_once_per_fresh_open(monkeypatch):
    _, sent = fake_iokit(monkeypatch)
    tr = ObsbotUSBTransport()
    tr.open()
    tr.open()  # already open: no second write
    tr.close()
    tr.open()  # a replugged camera boots with tracking back on
    assert [s[3] >> 8 for s in sent] == [2, 2]


def test_a_refused_tracking_write_still_leaves_pan_tilt_zoom(monkeypatch):
    _, sent = fake_iokit(monkeypatch, xu_ok=False)
    tr = ObsbotUSBTransport()
    tr.open()
    assert tr.tracking_error and "rejected" in tr.tracking_error
    tr.set_pantilt(0, 0)  # the standard UVC controls still go through
    assert sent[-1][3] == 0x0100


def test_no_obsbot_on_usb_names_the_ids_it_looked_for(monkeypatch):
    fake_iokit(monkeypatch, present=(0x2E1A, 0x4C04))  # an Insta360, not an OBSBOT
    with pytest.raises(Exception, match="no OBSBOT camera on USB.*3564:fef8, 3564:fef9"):
        ObsbotUSBTransport().open()


def test_tracking_payload_on_is_mode_two():
    assert tracking_payload(True)[:4] == bytes([0x16, 2, 2, 0])


def camera(zoom_range: tuple[int, int]) -> tuple[ObsbotCamera, FakeTransport]:
    tr = FakeTransport(fake_ranges=PTZRanges(pan=(-504000, 504000), tilt=(-252000, 108000), zoom=zoom_range))
    cam = ObsbotCamera("hero", "synthetic", transport=tr, tuning=Tuning(**FAST))
    return cam, tr


@pytest.mark.parametrize(("zoom_range", "expect"), [
    ((0, 100), {1.0: 0, 2.5: 50, 4.0: 100}),     # percent-style range
    ((100, 400), {1.0: 100, 2.5: 250, 4.0: 400}),  # Link-style range: factor x 100, as before
])
def test_zoom_maps_one_to_four_x_over_the_reported_range(zoom_range, expect):
    cam, tr = camera(zoom_range)
    for factor, raw in expect.items():
        cam.move(zoom=factor)
        assert tr.calls[-1] == ("zoom", raw)
    cam.move(zoom=8.0)
    assert cam.pose.zoom == 4.0 and tr.calls[-1] == ("zoom", zoom_range[1]), "clamped to the top"


def test_pan_and_tilt_are_arc_seconds_inside_the_reported_range():
    cam, tr = camera((0, 100))
    cam.move(pan=10.0, tilt=-5.0)
    assert ("pantilt", 36000, -18000) in tr.calls
    cam.move(pan=200.0, tilt=50.0)
    assert ("pantilt", 504000, 108000) in tr.calls, "clamped to what the gimbal reports"


def test_obsbot_knobs_live_under_their_own_prefix(monkeypatch):
    monkeypatch.setenv("OBSBOT_PID", "fefa")
    monkeypatch.setenv("OBSBOT_HFOV_DEG", "71")
    cam = ObsbotCamera("hero", "synthetic", transport=FakeTransport())
    assert cam.tuning.pids == (0xFEFA,) and cam.tuning.hfov_deg == 71.0
    assert Tuning.from_env().hfov_deg == 67.0, "INSTA360_* defaults are untouched"
    monkeypatch.delenv("OBSBOT_HFOV_DEG")
    assert ObsbotCamera("hero", "synthetic", transport=FakeTransport()).tuning.hfov_deg == 72.9


def test_backend_name_builds_a_working_synthetic_obsbot():
    cam = build_camera("hero", "obsbot", "synthetic")
    assert isinstance(cam, ObsbotCamera)
    assert cam.describe()["backend"] == "obsbot" and cam.describe()["ptz"] == "fake"
    cam.close()


def test_name_selection_picks_the_exact_obsbot_and_refuses_a_vague_one():
    devices = [(0, "FaceTime HD Camera"), (1, "OBSBOT Tiny 2 Lite"), (2, "OBSBOT Tiny 2"),
               (3, "OBSBOT Virtual Camera")]
    assert resolve_source("name:OBSBOT Tiny 2", devices) == "2", "exact name beats the Lite"
    with pytest.raises(CameraError, match="several cameras"):
        resolve_source("name:OBSBOT", devices)  # OBSBOT Center's virtual camera is there too


def _load_probe():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "obsbot_probe.py"
    spec = importlib.util.spec_from_file_location("obsbot_probe", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_finds_obsbots_in_the_usb_tree_including_an_unknown_model():
    probe = _load_probe()
    devices = [{"idVendor": 0x2E1A, "idProduct": 0x4C04, "USB Product Name": "Insta360 Link 2"},
               {"idVendor": 0x3564, "idProduct": 0xFEF8, "USB Product Name": "OBSBOT Tiny 2"},
               {"idVendor": 0x3564, "idProduct": 0xFF00, "kUSBProductString": "OBSBOT Tiny SE"}]
    assert probe.obsbots(devices) == [("OBSBOT Tiny 2", 0x3564, 0xFEF8),
                                      ("OBSBOT Tiny SE", 0x3564, 0xFF00)]


def test_probe_control_path_only_reads(monkeypatch):
    probe = _load_probe()
    _, sent = fake_iokit(monkeypatch, present=(OBSBOT_VID, 0xFEF8))
    tr = probe.control_path((0xFEF8, 0xFEF9))
    assert tr is not None and tr.pid == 0xFEF8
    assert sent and all(bm == 0xA1 for bm, *_ in sent), "no SET before --wiggle"
    tr.close()
