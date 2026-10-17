"""Insta360 Link 2 adapter (the gimbal model, USB 2e1a:4c04; not the gimbal-less Link 2C).

Control paths, in the order they are tried:
1. macOS: USB-direct UVC control transfers through link-ctl's in-process IOKit
   module (`link_usb_macos`). No Link Controller app, no sudo, no signing.
   link-ctl opens the original Link (PID 0x4c01) by default, so the Link 2 PID
   is passed explicitly.
2. Linux: v4l2-ctl pan_absolute / tilt_absolute / zoom_absolute.
The Link Controller WebSocket API is deliberately not used: its confirmed
message set has zoom, re-centre and joystick *velocity*, but no absolute
pan/tilt, so it cannot aim at a bbox.

Both paths drive standard UVC Camera Terminal controls (unit 1):
  CT_PANTILT_ABSOLUTE 0x0D  8 bytes: pan int32 LE, tilt int32 LE (arc-seconds per UVC spec)
  CT_ZOOM_ABSOLUTE    0x0B  2 bytes: uint16 LE, 100..400 on the Link family (1x..4x digital)
Ranges come from the camera (GET_MIN / GET_MAX); the fallbacks below are only
used if it will not report them.

Every PTZ command runs under a timeout. Any failure surfaces as CameraError,
which ToolBox turns into a tool error so the Director moves to the next camera.
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TypeVar

import numpy as np

from .base import Camera, CameraError, Pose, SyntheticSource, _FrameSource

log = logging.getLogger(__name__)

INSTA360_VID = 0x2E1A
# 4c04 = Link 2, 4c03 = Link 2 Pro (both gimbal), 4c01 = original Link.
# 4c02 (Link 2C, no gimbal) is left out on purpose.
GIMBAL_PIDS = (0x4C04, 0x4C03, 0x4C01)

CT_UNIT = 1
CT_ZOOM_ABSOLUTE = 0x0B
CT_PANTILT_ABSOLUTE = 0x0D
SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_RES, GET_LEN, GET_INFO, GET_DEF = (
    0x01, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87)
_REQ_IN, _REQ_OUT = 0xA1, 0x21  # class request to interface, device-to-host / host-to-device

T = TypeVar("T")


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_pids(key: str) -> tuple[int, ...]:
    raw = os.environ.get(key, "")
    return tuple(int(p, 16) for p in raw.split(",") if p.strip()) or GIMBAL_PIDS


@dataclass
class Tuning:
    """All knobs that may need adjusting on the real gimbal. Env names in comments."""

    transport: str = "auto"         # INSTA360_TRANSPORT: auto | usb | v4l2 | fake
    pids: tuple[int, ...] = GIMBAL_PIDS  # INSTA360_PID: hex list, e.g. "4c04"
    v4l2_device: str = ""           # INSTA360_V4L2_DEVICE, default /dev/video<source>
    hfov_deg: float = 67.0          # INSTA360_HFOV_DEG: Insta360 spec sheet, 1x zoom
    fill: float = 0.6               # INSTA360_FILL: share of the frame the target should fill
    max_zoom: float = 4.0           # INSTA360_MAX_ZOOM: cap below the 4x digital limit if crops get soft
    units_per_deg: float = 3600.0   # INSTA360_UNITS_PER_DEG: UVC arc-seconds
    pan_sign: float = 1.0           # INSTA360_PAN_SIGN: -1 if the gimbal pans the wrong way
    tilt_sign: float = 1.0          # INSTA360_TILT_SIGN: -1 if it tilts the wrong way
    settle_s: float = 0.35          # INSTA360_SETTLE_S: fixed wait after any move (stabilise + AF)
    slew_deg_s: float = 120.0       # INSTA360_SLEW_DEG_S: assumed gimbal speed
    zoom_settle_s: float = 0.1      # INSTA360_ZOOM_SETTLE_S: extra wait per 1x of zoom change
    home_after_s: float = 6.0       # INSTA360_HOME_AFTER_S: return to the wide view when idle; 0 = never
    home_pan: float = 0.0           # INSTA360_HOME_PAN
    home_tilt: float = 0.0          # INSTA360_HOME_TILT
    cmd_timeout_s: float = 2.0      # INSTA360_CMD_TIMEOUT_S: gimbal silent this long = failed

    @classmethod
    def from_env(cls) -> Tuning:
        d = cls()
        return cls(
            transport=os.environ.get("INSTA360_TRANSPORT", d.transport),
            pids=_env_pids("INSTA360_PID"),
            v4l2_device=os.environ.get("INSTA360_V4L2_DEVICE", d.v4l2_device),
            hfov_deg=_env_float("INSTA360_HFOV_DEG", d.hfov_deg),
            fill=_env_float("INSTA360_FILL", d.fill),
            max_zoom=_env_float("INSTA360_MAX_ZOOM", d.max_zoom),
            units_per_deg=_env_float("INSTA360_UNITS_PER_DEG", d.units_per_deg),
            pan_sign=_env_float("INSTA360_PAN_SIGN", d.pan_sign),
            tilt_sign=_env_float("INSTA360_TILT_SIGN", d.tilt_sign),
            settle_s=_env_float("INSTA360_SETTLE_S", d.settle_s),
            slew_deg_s=_env_float("INSTA360_SLEW_DEG_S", d.slew_deg_s),
            zoom_settle_s=_env_float("INSTA360_ZOOM_SETTLE_S", d.zoom_settle_s),
            home_after_s=_env_float("INSTA360_HOME_AFTER_S", d.home_after_s),
            home_pan=_env_float("INSTA360_HOME_PAN", d.home_pan),
            home_tilt=_env_float("INSTA360_HOME_TILT", d.home_tilt),
            cmd_timeout_s=_env_float("INSTA360_CMD_TIMEOUT_S", d.cmd_timeout_s),
        )


@dataclass
class PTZRanges:
    """Raw device units, exactly as the camera reports them."""

    pan: tuple[int, int]
    tilt: tuple[int, int]
    zoom: tuple[int, int]


# Same envelope as the Director's `move` tool schema; zoom 100..400 is the Link
# family's documented CT_ZOOM range.
FALLBACK_RANGES = PTZRanges(pan=(-140 * 3600, 140 * 3600), tilt=(-70 * 3600, 30 * 3600),
                            zoom=(100, 400))


# --- geometry ----------------------------------------------------------------------

def aim_at(bbox: tuple[int, int, int, int], frame_wh: tuple[int, int], pose: Pose,
           hfov_deg: float) -> tuple[float, float]:
    """Absolute (pan, tilt) in degrees that centres `bbox`, seen in a frame taken at
    `pose`. Pinhole model; digital zoom narrows the FOV in tan-space. Exact for a
    pan-then-tilt gimbal, so it stays right when the camera is already tilted."""
    x1, y1, x2, y2 = bbox
    w, h = frame_wh
    half_w = math.tan(math.radians(hfov_deg) / 2) / max(pose.zoom, 1e-6)
    half_h = half_w * h / w  # square pixels
    # Ray through the bbox centre in camera coords: x right, y up, z forward.
    rx = ((x1 + x2) / 2 - w / 2) / (w / 2) * half_w
    ry = -((y1 + y2) / 2 - h / 2) / (h / 2) * half_h
    rz = 1.0
    t, p = math.radians(pose.tilt), math.radians(pose.pan)
    ry, rz = ry * math.cos(t) + rz * math.sin(t), -ry * math.sin(t) + rz * math.cos(t)
    rx, rz = rx * math.cos(p) + rz * math.sin(p), -rx * math.sin(p) + rz * math.cos(p)
    return math.degrees(math.atan2(rx, rz)), math.degrees(math.atan2(ry, math.hypot(rx, rz)))


def zoom_for(bbox: tuple[int, int, int, int], frame_wh: tuple[int, int], current_zoom: float,
             fill: float) -> float:
    """Zoom at which the bbox's tighter dimension fills `fill` of the frame.
    Unclamped; the camera clamps to what the hardware allows."""
    x1, y1, x2, y2 = bbox
    w, h = frame_wh
    return fill * current_zoom * min(w / max(x2 - x1, 1), h / max(y2 - y1, 1))


# --- transports --------------------------------------------------------------------

class TransportError(RuntimeError):
    """Raised by transports; Insta360Camera re-raises it as CameraError."""


class PTZTransport(ABC):
    name: str = "?"

    @abstractmethod
    def open(self) -> None:
        """Find the camera. Raise TransportError if it is not there."""

    @abstractmethod
    def ranges(self) -> PTZRanges | None:
        """Hardware limits in raw units, or None if the camera will not say."""

    @abstractmethod
    def set_pantilt(self, pan: int, tilt: int) -> None: ...

    @abstractmethod
    def set_zoom(self, zoom: int) -> None: ...

    def close(self) -> None:
        """Drop any handle so the next open() starts clean (handles unplug/replug)."""


class MacUSBTransport(PTZTransport):
    """UVC control transfers over IOKit via link-ctl's `link_usb_macos`."""

    name = "usb-iokit"

    def __init__(self, pids: tuple[int, ...] = GIMBAL_PIDS, ct_unit: int = CT_UNIT, iface: int = 0):
        self.pids, self.ct_unit, self.iface = pids, ct_unit, iface
        self.pid: int | None = None
        self._usb = None
        self._h = None

    def open(self) -> None:
        if self._h is not None:
            return
        try:
            import link_usb_macos as usb  # from the link-ctl package; macOS only
        except ImportError as e:
            raise TransportError(f"link-ctl USB backend unavailable ({e}); uv pip install link-ctl") from e
        for pid in self.pids:
            h = usb.open_vc_interface(vid=INSTA360_VID, pid=pid, iface_num=self.iface)
            if h is not None:
                self._usb, self._h, self.pid = usb, h, pid
                return
        want = ", ".join(f"{INSTA360_VID:04x}:{p:04x}" for p in self.pids)
        raise TransportError(f"no Insta360 gimbal on USB (looked for {want})")

    def request(self, req: int, sel: int, length: int, unit: int | None = None) -> bytes:
        """Raw UVC GET_* request; public so the probe can read anything."""
        self.open()
        unit = self.ct_unit if unit is None else unit
        ok, data = self._usb.control_request(self._h, _REQ_IN, req, sel << 8,
                                             (unit << 8) | self.iface, read_length=length)
        if not ok or len(data) < length:
            raise TransportError(f"UVC request 0x{req:02x} unit {unit} sel 0x{sel:02x} failed")
        return data

    def _set(self, sel: int, payload: bytes) -> None:
        self.open()
        ok, _ = self._usb.control_request(self._h, _REQ_OUT, SET_CUR, sel << 8,
                                          (self.ct_unit << 8) | self.iface, data=payload)
        if not ok:
            self.close()
            raise TransportError(f"UVC SET_CUR sel 0x{sel:02x} rejected")

    def ranges(self) -> PTZRanges | None:
        try:
            pmin, tmin = struct.unpack("<ii", self.request(GET_MIN, CT_PANTILT_ABSOLUTE, 8))
            pmax, tmax = struct.unpack("<ii", self.request(GET_MAX, CT_PANTILT_ABSOLUTE, 8))
            (zmin,) = struct.unpack("<H", self.request(GET_MIN, CT_ZOOM_ABSOLUTE, 2))
            (zmax,) = struct.unpack("<H", self.request(GET_MAX, CT_ZOOM_ABSOLUTE, 2))
        except TransportError as e:
            log.warning("Insta360 did not report PTZ ranges (%s); using fallbacks", e)
            return None
        return PTZRanges(pan=(pmin, pmax), tilt=(tmin, tmax), zoom=(zmin, zmax))

    def set_pantilt(self, pan: int, tilt: int) -> None:
        self._set(CT_PANTILT_ABSOLUTE, struct.pack("<ii", pan, tilt))

    def set_zoom(self, zoom: int) -> None:
        self._set(CT_ZOOM_ABSOLUTE, struct.pack("<H", zoom))

    def close(self) -> None:
        if self._h is not None:
            self._h.close()
        self._h = None


_V4L2_CTRL = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\(\w+\)\s*:\s*(.*)$")


def parse_v4l2_ctrls(text: str) -> dict[str, dict[str, int]]:
    """`v4l2-ctl --list-ctrls` lines like
    `pan_absolute 0x009a0908 (int) : min=-522000 max=522000 step=3600 default=0 value=0`."""
    out: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        m = _V4L2_CTRL.match(line)
        if m:
            out[m.group(1)] = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(2))}
    return out


class V4L2Transport(PTZTransport):
    name = "v4l2"

    def __init__(self, device: str):
        self.device = device
        self.ctrls: dict[str, dict[str, int]] = {}

    def _run(self, *args: str) -> str:
        try:
            r = subprocess.run(["v4l2-ctl", "-d", self.device, *args],
                               check=True, capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise TransportError(f"v4l2-ctl {' '.join(args)} failed: {e}") from e
        return r.stdout

    def open(self) -> None:
        if not shutil.which("v4l2-ctl"):
            raise TransportError("v4l2-ctl not found (install v4l-utils)")
        self.ctrls = parse_v4l2_ctrls(self._run("--list-ctrls"))
        if "pan_absolute" not in self.ctrls:
            raise TransportError(f"{self.device} has no pan_absolute control")

    def ranges(self) -> PTZRanges | None:
        try:
            c = self.ctrls
            return PTZRanges(pan=(c["pan_absolute"]["min"], c["pan_absolute"]["max"]),
                             tilt=(c["tilt_absolute"]["min"], c["tilt_absolute"]["max"]),
                             zoom=(c["zoom_absolute"]["min"], c["zoom_absolute"]["max"]))
        except KeyError:
            return None

    def set_pantilt(self, pan: int, tilt: int) -> None:
        self._run("--set-ctrl", f"pan_absolute={pan},tilt_absolute={tilt}")

    def set_zoom(self, zoom: int) -> None:
        self._run("--set-ctrl", f"zoom_absolute={zoom}")


@dataclass
class FakeTransport(PTZTransport):
    """Records commands. Used by tests and `--synthetic` runs. Set `fail` to make
    the gimbal refuse, or `hang_s` to make it go silent."""

    name: str = "fake"
    fake_ranges: PTZRanges | None = field(default_factory=lambda: FALLBACK_RANGES)
    fail: bool = False
    hang_s: float = 0.0
    calls: list[tuple] = field(default_factory=list)

    def _io(self) -> None:
        if self.hang_s:
            time.sleep(self.hang_s)
        if self.fail:
            raise TransportError("simulated: gimbal not responding")

    def open(self) -> None:
        self._io()

    def ranges(self) -> PTZRanges | None:
        return self.fake_ranges

    def set_pantilt(self, pan: int, tilt: int) -> None:
        self._io()
        self.calls.append(("pantilt", pan, tilt))

    def set_zoom(self, zoom: int) -> None:
        self._io()
        self.calls.append(("zoom", zoom))


@dataclass
class UnavailableTransport(PTZTransport):
    reason: str
    name: str = "none"

    def open(self) -> None:
        raise TransportError(self.reason)

    def ranges(self) -> PTZRanges | None:
        return None

    def set_pantilt(self, pan: int, tilt: int) -> None:
        raise TransportError(self.reason)

    def set_zoom(self, zoom: int) -> None:
        raise TransportError(self.reason)


def make_transport(source: str, tuning: Tuning) -> PTZTransport:
    kind = tuning.transport
    if kind == "auto":
        kind = "usb" if sys.platform == "darwin" else "v4l2" if sys.platform.startswith("linux") else ""
    if kind == "usb":
        return MacUSBTransport(tuning.pids)
    if kind == "v4l2":
        return V4L2Transport(tuning.v4l2_device or f"/dev/video{source}")
    if kind == "fake":
        return FakeTransport()
    return UnavailableTransport(f"no PTZ control path for {sys.platform!r} (INSTA360_TRANSPORT={tuning.transport})")


# --- camera ------------------------------------------------------------------------

class Insta360Camera(Camera):
    """Real optics: the stream already is the aimed, zoomed view, so snapshot()
    never crops. wide_frame() is the live view at whatever pose the gimbal holds;
    it returns home on its own after `home_after_s` idle seconds."""

    def __init__(self, name: str, source: str | None, transport: PTZTransport | None = None,
                 tuning: Tuning | None = None, frames=None):
        self.name = name
        self.tuning = tuning or Tuning.from_env()
        synthetic = source in (None, "synthetic")
        self.src = frames or (SyntheticSource() if synthetic else _FrameSource(str(source)))
        if transport is None:
            transport = FakeTransport() if synthetic else make_transport(str(source), self.tuning)
        self.transport = transport
        self.ranges = FALLBACK_RANGES
        self.home = Pose(self.tuning.home_pan, self.tuning.home_tilt, 1.0)
        self.pose = replace(self.home)
        self.ptz_error: str | None = None
        self._connected = False
        self._lock = threading.RLock()
        self._settle_until = 0.0
        self._home_timer: threading.Timer | None = None
        self._home_gen = 0
        # Pose and size of the last frame handed to the Scout: detection bboxes
        # are in that frame, even if the gimbal has moved since.
        self._view_pose = replace(self.home)
        self._view_wh: tuple[int, int] | None = None
        try:
            with self._lock:
                self._drive(self.home, force=True)
        except CameraError as e:  # keep running; the Director sees the error per call
            log.warning("%s", e)

    # --- Camera interface ----------------------------------------------------------
    def wide_frame(self) -> np.ndarray:
        self._check()
        pose = replace(self.pose)  # no lock: a hung gimbal must not stall the Scout
        f = self.src.read()
        self._view_pose, self._view_wh = pose, (f.shape[1], f.shape[0])
        return f

    def snapshot(self) -> np.ndarray:
        self._check()
        wait = self._settle_until - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        f = self.src.read(newer_than=self._settle_until)
        self._arm_home()
        return f

    def move(self, pan: float | None = None, tilt: float | None = None,
             zoom: float | None = None) -> Pose:
        self._check()
        with self._lock:
            target = Pose(self.pose.pan if pan is None else float(pan),
                          self.pose.tilt if tilt is None else float(tilt),
                          self.pose.zoom if zoom is None else float(zoom))
            self._drive(target)
            self._arm_home()
            return replace(self.pose)

    def look_at(self, bbox: tuple[int, int, int, int]) -> Pose:
        self._check()
        if self._view_wh is None:
            self.wide_frame()
        assert self._view_wh is not None
        with self._lock:
            pan, tilt = aim_at(bbox, self._view_wh, self._view_pose, self.tuning.hfov_deg)
            zoom = zoom_for(bbox, self._view_wh, self._view_pose.zoom, self.tuning.fill)
            return self.move(pan, tilt, zoom)

    def describe(self) -> dict:
        d = super().describe()
        d.update(backend="insta360",
                 ptz=self.transport.name if self._connected else f"offline: {self.ptz_error}")
        return d

    def close(self) -> None:
        self._cancel_home()
        self.transport.close()

    # --- internals -----------------------------------------------------------------
    def _limits(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """Hardware ranges as degrees (pan, tilt) and the max zoom factor."""
        u, r = self.tuning.units_per_deg, self.ranges

        def deg(raw: tuple[int, int], sign: float) -> tuple[float, float]:
            a, b = raw[0] * sign / u, raw[1] * sign / u
            return min(a, b), max(a, b)

        zmax = min(self.tuning.max_zoom, r.zoom[1] / self._zoom_1x())
        return deg(r.pan, self.tuning.pan_sign), deg(r.tilt, self.tuning.tilt_sign), zmax

    def _zoom_1x(self) -> int:
        """Raw zoom value for 1x: the reported minimum (100 on the Link family)."""
        return self.ranges.zoom[0] if self.ranges.zoom[0] > 0 else 100

    def _io(self, fn: Callable[[], T]) -> T:
        """Run one transport call, giving up after cmd_timeout_s. A hung USB
        transfer must not freeze the Director, so it runs on a daemon thread."""
        box: dict = {}

        def run() -> None:
            try:
                box["ok"] = fn()
            except Exception as e:  # noqa: BLE001 - re-raised in the caller below
                box["err"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(self.tuning.cmd_timeout_s)
        if t.is_alive():
            raise TransportError(f"no answer within {self.tuning.cmd_timeout_s:.1f}s")
        if "err" in box:
            raise box["err"]
        return box.get("ok")

    def _drive(self, target: Pose, force: bool = False) -> None:
        """Send `target` to the gimbal (caller holds the lock). Clamps to hardware
        limits, records the settle deadline, raises CameraError on any failure."""
        tn = self.tuning
        try:
            if not self._connected:
                self._io(self.transport.open)
                self.ranges = self._io(self.transport.ranges) or FALLBACK_RANGES
                self._connected = True
            (p0, p1), (t0, t1), zmax = self._limits()
            target = Pose(min(max(target.pan, p0), p1), min(max(target.tilt, t0), t1),
                          min(max(target.zoom, 1.0), zmax))
            if force or (target.pan, target.tilt) != (self.pose.pan, self.pose.tilt):
                pan_raw = round(target.pan * tn.units_per_deg * tn.pan_sign)
                tilt_raw = round(target.tilt * tn.units_per_deg * tn.tilt_sign)
                self._io(lambda: self.transport.set_pantilt(pan_raw, tilt_raw))
            if force or target.zoom != self.pose.zoom:
                zoom_raw = round(target.zoom * self._zoom_1x())
                self._io(lambda: self.transport.set_zoom(zoom_raw))
        except (TransportError, OSError) as e:
            self._connected = False
            self.ptz_error = str(e)
            try:
                self.transport.close()
            except (TransportError, OSError):
                pass
            raise CameraError(f"{self.name}: gimbal not responding ({e})") from e
        travel = max(abs(target.pan - self.pose.pan), abs(target.tilt - self.pose.tilt))
        settle = (tn.settle_s + travel / max(tn.slew_deg_s, 1e-6)
                  + abs(target.zoom - self.pose.zoom) * tn.zoom_settle_s)
        self._settle_until = max(self._settle_until, time.monotonic() + settle)
        self.pose, self.ptz_error = target, None

    def _arm_home(self) -> None:
        self._cancel_home()
        if self.tuning.home_after_s <= 0 or self.pose == self.home:
            return
        self._home_timer = threading.Timer(self.tuning.home_after_s, self._go_home,
                                           args=(self._home_gen,))
        self._home_timer.daemon = True
        self._home_timer.start()

    def _cancel_home(self) -> None:
        # Bumping the generation also defuses a timer that already fired and is
        # waiting on the lock; cancel() alone cannot stop that one.
        self._home_gen += 1
        if self._home_timer is not None:
            self._home_timer.cancel()
            self._home_timer = None

    def _go_home(self, gen: int) -> None:
        with self._lock:
            if gen != self._home_gen:
                return
            try:
                self._drive(self.home)
            except CameraError as e:
                log.warning("%s", e)
