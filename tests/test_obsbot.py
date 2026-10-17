"""The OBSBOT adapter against a fake IOKit module and a fake gimbal. None of this has run
on a real OBSBOT yet; scripts/obsbot_probe.py is the hardware check."""
import struct
import sys
import types

import pytest
from sherlock.camera import ObsbotCamera, build_camera, obsbot
from sherlock.camera.base import CameraError, resolve_source
from sherlock.camera.insta360 import SET_CUR, FakeTransport, PTZRanges, Tuning
from sherlock.camera.obsbot import (
    OBSBOT_VID,
    WARN_DRIFT,
    WARN_TRACKING_ON,
    ObsbotUSBTransport,
    gesture_payload,
    make_obsbot_transport,
    tracking_payload,
)

FAST = {"home_after_s": 0.0, "settle_s": 0.0, "cmd_timeout_s": 1.0}


def fake_iokit(monkeypatch, present=(OBSBOT_VID, 0xFEF9), xu_ok=True):
    """link_usb_macos with one camera on the bus; records every control transfer."""
    opened, sent = [], []

    def open_vc_interface(vid, pid, iface_num):
        key = (vid, pid) if iface_num == 0 else (vid, pid, iface_num)
        opened.append(key)
        return types.SimpleNamespace(close=lambda: None) if key == present else None

    def control_request(handle, bm, req, w_value, w_index, data=b"", read_length=0):
        sent.append((bm, req, w_value, w_index, bytes(data)))
        return (xu_ok or w_index >> 8 != 2), b"\0" * read_length

    fake = types.ModuleType("link_usb_macos")
    fake.open_vc_interface, fake.control_request = open_vc_interface, control_request
    monkeypatch.setitem(sys.modules, "link_usb_macos", fake)
    return opened, sent


def test_opens_the_obsbot_by_its_ids_and_switches_tracking_and_gestures_off_first(monkeypatch):
    opened, sent = fake_iokit(monkeypatch)
    tr = ObsbotUSBTransport()
    tr.open()
    assert opened == [(0x3564, 0xFEF8), (0x3564, 0xFEF9)] and tr.pid == 0xFEF9
    for (bm, req, w_value, w_index, payload), head in zip(sent[:2], ([0x16, 2, 0, 0], [0x20, 5, 0, 0]),
                                                          strict=True):
        assert (bm, req, w_value, w_index) == (0x21, SET_CUR, 0x06 << 8, 2 << 8), "XU 2, selector 6"
        assert len(payload) == 60 and payload[:4] == bytes(head) and not any(payload[2:])
    tr.set_pantilt(36000, -7200)
    tr.set_zoom(50)
    assert sent[2:] == [(0x21, SET_CUR, 0x0D00, 0x0100, struct.pack("<ii", 36000, -7200)),
                        (0x21, SET_CUR, 0x0B00, 0x0100, struct.pack("<H", 50))]


def test_tracking_and_gestures_are_switched_off_once_per_fresh_open(monkeypatch):
    _, sent = fake_iokit(monkeypatch)
    tr = ObsbotUSBTransport()
    tr.open()
    tr.open()  # already open: no second write
    tr.close()
    tr.open()  # a replugged camera boots with tracking back on
    assert [s[3] >> 8 for s in sent] == [2, 2, 2, 2]


def test_a_refused_tracking_write_still_leaves_pan_tilt_zoom(monkeypatch):
    _, sent = fake_iokit(monkeypatch, xu_ok=False)
    tr = ObsbotUSBTransport()
    tr.open()
    assert tr.assist_error == "could not switch AI tracking or gesture control off"
    tr.set_pantilt(0, 0)  # the standard UVC controls still go through
    assert sent[-1][3] == 0x0100


def test_no_obsbot_on_usb_names_the_ids_it_looked_for(monkeypatch):
    fake_iokit(monkeypatch, present=(0x2E1A, 0x4C04))  # an Insta360, not an OBSBOT
    with pytest.raises(Exception, match="no OBSBOT camera on USB.*3564:fef8, 3564:fef9"):
        ObsbotUSBTransport().open()


def test_switch_on_payloads():
    assert tracking_payload(True)[:4] == bytes([0x16, 2, 2, 0])
    assert gesture_payload(True)[:7] == bytes([0x20, 5, 1, 0, 0, 0, 0])


def test_unit_ids_and_interface_come_from_the_environment(monkeypatch):
    """The Tiny 2 Lite's ids are not published: the probe finds them, env sets them."""
    opened, sent = fake_iokit(monkeypatch, present=(OBSBOT_VID, 0xFEF9, 1))
    for key, value in (("OBSBOT_CT_UNIT", "3"), ("OBSBOT_XU_UNIT", "0x04"), ("OBSBOT_IFACE", "1")):
        monkeypatch.setenv(key, value)
    tr = make_obsbot_transport("0", Tuning.from_env("OBSBOT", obsbot.DEFAULTS))
    tr.open()
    tr.set_zoom(0)
    assert opened[-1] == (OBSBOT_VID, 0xFEF9, 1)
    assert [s[3] for s in sent] == [0x0401, 0x0401, 0x0301]


def test_switch_off_is_repeated_once_the_stream_flows(monkeypatch):
    _, sent = fake_iokit(monkeypatch)
    cam = ObsbotCamera("hero", "synthetic", transport=ObsbotUSBTransport(), tuning=Tuning(**FAST))
    assert [s[4][0] for s in sent if s[3] >> 8 == 2] == [0x16, 0x20, 0x16, 0x20]
    assert "warning" not in cam.describe()
    cam.close()


def test_a_refused_switch_off_and_a_dead_gimbal_are_labelled(monkeypatch):
    fake_iokit(monkeypatch, xu_ok=False)
    cam = ObsbotCamera("hero", "synthetic", transport=ObsbotUSBTransport(), tuning=Tuning(**FAST))
    d = cam.describe()
    assert d["warning"] == WARN_TRACKING_ON and "gesture control" in d["tracking"]
    cam.close()
    dead = ObsbotCamera("hero", "synthetic", transport=FakeTransport(fail=True), tuning=Tuning(**FAST))
    assert dead.describe()["warning"] == "PTZ OFFLINE"
    dead.close()


class DriftingGimbal(FakeTransport):
    live = (0, 0, 0)

    def live_ptz(self):
        return self.live


@pytest.mark.parametrize(("samples", "flagged"), [
    ([(36000, 0, 0), (36000, 0, 0)], True),    # 10 deg off, twice: something else moved it
    ([(36000, 0, 0)], False),                  # once is not enough
    ([(36000, 0, 0), (18000, 0, 0)], False),   # closing in: a slow move, not tracking
    ([(0, 0, 50), (0, 0, 50)], True),          # zoomed to 2.5x on its own (a gesture)
    ([(3600, 0, 0), (3600, 0, 0)], False),     # 1 deg: within tolerance
])
def test_a_gimbal_moving_on_its_own_is_reported(monkeypatch, samples, flagged):
    monkeypatch.setattr(obsbot, "DRIFT_GRACE_S", 0.0)
    monkeypatch.setattr(obsbot, "DRIFT_EVERY_S", 0.0)
    tr = DriftingGimbal(fake_ranges=PTZRanges(pan=(-504000, 504000), tilt=(-252000, 108000), zoom=(0, 100)))
    cam = ObsbotCamera("hero", "synthetic", transport=tr, tuning=Tuning(**FAST))
    for live in samples:
        tr.live = live
        cam.wide_frame()
    assert (cam.describe().get("warning") == WARN_DRIFT) is flagged
    tr.live = (0, 0, 0)
    cam.wide_frame()
    assert "warning" not in cam.describe(), "back on its aim: the label clears"
    cam.close()


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


def test_probe_finds_obsbots_in_the_usb_tree_and_the_hub_above_one():
    probe = _load_probe()
    tiny2 = {"idVendor": 0x3564, "idProduct": 0xFEF8, "USB Product Name": "OBSBOT Tiny 2"}
    se = {"idVendor": 0x3564, "idProduct": 0xFF00, "kUSBProductString": "OBSBOT Tiny SE"}
    hub = {"idVendor": 0x05E3, "idProduct": 0x0610, "bDeviceClass": 9, "USB Product Name": "USB2.1 Hub",
           "IORegistryEntryChildren": [se]}
    link = {"idVendor": 0x2E1A, "idProduct": 0x4C04, "USB Product Name": "Insta360 Link 2"}
    tree = {"IORegistryEntryChildren": [{"IORegistryEntryName": "AppleT6050USBXHCI",
                                         "IORegistryEntryChildren": [link, tiny2, hub]}]}
    assert probe.obsbots(tree) == [("OBSBOT Tiny 2", 0x3564, 0xFEF8, ()),
                                   ("OBSBOT Tiny SE", 0x3564, 0xFF00, ("USB2.1 Hub",))]


def test_probe_control_path_only_reads(monkeypatch):
    probe = _load_probe()
    _, sent = fake_iokit(monkeypatch, present=(OBSBOT_VID, 0xFEF8))
    tr = probe.control_path((0xFEF8, 0xFEF9))
    assert tr is not None and tr.pid == 0xFEF8 and (tr.iface, tr.ct_unit) == (0, 1)
    assert sent and all(bm == 0xA1 for bm, *_ in sent), "no SET before --wiggle"
    tr.close()
