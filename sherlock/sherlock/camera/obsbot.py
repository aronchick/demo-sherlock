"""OBSBOT adapter, hardware-tested on the Tiny 2 Lite. See docs/OBSBOT_HARDWARE.md.

The OBSBOT Tiny family is a UVC camera on a two-axis gimbal, so this is the Insta360
adapter (same IOKit control transfers through link-ctl, same aiming, settling and
fault handling) with OBSBOT's USB ids, specs and three differences:

- AI tracking and gesture control both move the gimbal on their own and would fight
  every look_at, so both are switched off through OBSBOT's vendor extension unit when
  the camera connects, and again once its stream is flowing.
- The live gimbal position is read back now and then; a gimbal that moves on its own
  anyway is reported in describe() and on the board instead of failing silently.
- Zoom is mapped over whatever raw range the camera reports, instead of the Link
  family's raw = factor x 100.

Facts, with sources (checked 2026-10-17):
- USB ids: vendor 0x3564 (Remo Tech). Tiny 2 = fef8: Linux kernel patch thread
  "uvcvideo: fix up missing AUTO_UPDATE on OBSBOT Tiny 2 pan/tilt" (lkml, 2026-08/09,
  full lsusb -v), plus three Linux control projects. Tiny 2 Lite = fef9: lsusb on the
  openSUSE forum and Reddit, and the device tables of Domatix/obsbot-control and
  samuelcolvin/obsbot-rs. Tiny SE: no public source found; read it with
  scripts/obsbot_probe.py and pass OBSBOT_PID=<hex>.
- Unit ids: on the Tiny 2 the Camera Terminal is unit 1 (PanTilt and Zoom Absolute)
  and the extension unit is unit 2 (lkml lsusb -v). No public descriptor dump of the
  Tiny 2 Lite was found; obsbot-rs reads its ids from the descriptors at runtime. The
  probe scans for them; override with OBSBOT_CT_UNIT / OBSBOT_XU_UNIT / OBSBOT_IFACE.
- Extension unit 2, selector 6 takes 60-byte [tag, len, values..., 0...] writes.
  AI mode tag 0x16 with (mode, sub-mode), 0 = off: cgevans/tiny2 (EUPL-1.2, Tiny 2) and
  samuelcolvin/obsbot-rs (MIT, Tiny 2 Lite) agree. Gesture control tag 0x20 with five
  bytes (enable, mode, zoom flag, u16 zoom ratio): obsbot-rs. Neither is from OBSBOT;
  the official route is OBSBOT Center or its SDK (sign-up required).
- Specs (obsbot.com tech specs): pan +-140 deg, tilt 30 deg down to 70 deg up,
  120 deg/s, zoom 1-4x; horizontal FOV 72.9 deg (Tiny 2), 71 deg (Tiny SE).
- The camera sleeps after about a minute without a stream and then silently ignores
  writes (obsbot-rs); Sherlock keeps the stream open, so the gimbal should stay awake.

Without hardware control (no link-ctl, Linux, or a model that refuses), use the
software crop on the same stream instead: CAMERAS="hero:fake:name:OBSBOT Tiny 2".
"""
from __future__ import annotations

import logging
import os
import struct
import sys
import time

from .base import CameraError
from .insta360 import (
    CT_PANTILT_ABSOLUTE,
    CT_UNIT,
    CT_ZOOM_ABSOLUTE,
    GET_CUR,
    FakeTransport,
    Insta360Camera,
    MacUSBTransport,
    PTZTransport,
    TransportError,
    Tuning,
    UnavailableTransport,
    V4L2Transport,
)

log = logging.getLogger(__name__)

OBSBOT_VID = 0x3564
OBSBOT_PIDS = (0xFEF8, 0xFEF9)  # Tiny 2, Tiny 2 Lite
XU_UNIT, XU_SEL_SIMPLE, XU_LEN = 2, 0x06, 60
TAG_AI_MODE, TAG_GESTURE = 0x16, 0x20

# Tiny 2 Lite hardware: positive UVC pan turns left; 1x -> 2x takes ~2 s.
DEFAULTS = Tuning(pids=OBSBOT_PIDS, hfov_deg=72.9, max_zoom=4.0, slew_deg_s=120.0,
                 pan_sign=-1.0, zoom_settle_s=2.2)
# Zoom factor at the camera's raw maximum: the spec sheet's 4x.
ZOOM_AT_MAX = 4.0

# The gimbal is moving on its own when, this long after a move should have settled,
# its live position is this far off the commanded one and not closing in. The grace
# covers a gimbal slower than slew_deg_s assumes (lkml: Tiny 2 moves "take seconds").
DRIFT_DEG, DRIFT_ZOOM, DRIFT_GRACE_S, DRIFT_EVERY_S = 2.0, 0.25, 3.0, 1.0

# describe()["warning"]: the short label the board shows on the hero tile.
WARN_TRACKING_ON, WARN_DRIFT = "TRACKING NOT SWITCHED OFF", "GIMBAL MOVING ON ITS OWN"


def tag_payload(tag: int, values: bytes) -> bytes:
    """One write to the extension unit's simple channel."""
    return bytes([tag, len(values), *values]).ljust(XU_LEN, b"\0")


def tracking_payload(on: bool) -> bytes:
    return tag_payload(TAG_AI_MODE, bytes([2 if on else 0, 0]))


def gesture_payload(on: bool) -> bytes:
    return tag_payload(TAG_GESTURE, bytes([int(on), 0, 0, 0, 0]))


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    return int(v, 0) if v.strip() else default


class ObsbotUSBTransport(MacUSBTransport):
    """link-ctl's IOKit control transfers, aimed at an OBSBOT. Switches AI tracking and
    gesture control off on every fresh open, since a replugged camera boots with them
    as OBSBOT Center left them."""

    name = "obsbot-usb"

    def __init__(self, pids: tuple[int, ...] = OBSBOT_PIDS, assist_off: bool = True,
                 ct_unit: int = CT_UNIT, xu_unit: int = XU_UNIT, iface: int = 0):
        super().__init__(pids, ct_unit=ct_unit, iface=iface, vid=OBSBOT_VID, label="OBSBOT camera")
        self.xu_unit = xu_unit
        self.assist_off = assist_off
        self.assist_error: str | None = None

    def open(self) -> None:
        fresh = self._h is None
        super().open()
        if fresh and self.assist_off:
            self.switch_assist_off()

    def switch_assist_off(self) -> None:
        """Turn AI tracking and gesture control off. A refusal is recorded, not raised:
        pan, tilt and zoom are standard UVC and still work."""
        failed = []
        for what, payload in (("AI tracking", tracking_payload(False)),
                              ("gesture control", gesture_payload(False))):
            try:
                self._set(XU_SEL_SIMPLE, payload, unit=self.xu_unit)
            except TransportError:
                failed.append(what)
        self.assist_error = f"could not switch {' or '.join(failed)} off" if failed else None
        if failed:
            log.warning("OBSBOT: %s; switch it off in OBSBOT Center, then quit it", self.assist_error)

    def set_tracking(self, on: bool) -> None:
        self._set(XU_SEL_SIMPLE, tracking_payload(on), unit=self.xu_unit)

    def set_pantilt(self, pan: int, tilt: int) -> None:
        """Verify motion and retry the Lite's stale-axis response after it settles."""
        super().set_pantilt(pan, tilt)
        deadline = time.monotonic() + 1.7
        retry_at = time.monotonic() + 0.5
        matched = 0
        while time.monotonic() < deadline:
            actual = struct.unpack("<ii", self.request(GET_CUR, CT_PANTILT_ABSOLUTE, 8))
            # Lite firmware truncates fractional degrees and biases negative
            # angles toward zero (e.g. -14.49 requested -> -13 reported).
            tolerance = int(DRIFT_DEG * 3600)
            matched = matched + 1 if abs(actual[0] - pan) <= tolerance and abs(actual[1] - tilt) <= tolerance else 0
            if matched >= 2:
                return
            if time.monotonic() >= retry_at:
                super().set_pantilt(pan, tilt)
                retry_at = time.monotonic() + 0.5
            time.sleep(0.05)
        raise TransportError(f"OBSBOT pan/tilt did not reach {(pan, tilt)}; read back {actual}")

    def live_ptz(self) -> tuple[int, int, int]:
        """Where the gimbal is now (the Tiny 2 reports its live position on GET_CUR)."""
        pan, tilt = struct.unpack("<ii", self.request(GET_CUR, CT_PANTILT_ABSOLUTE, 8))
        (zoom,) = struct.unpack("<H", self.request(GET_CUR, CT_ZOOM_ABSOLUTE, 2))
        return pan, tilt, zoom


def make_obsbot_transport(source: str, tuning: Tuning) -> PTZTransport:
    kind = tuning.transport
    if kind == "auto":
        kind = "usb" if sys.platform == "darwin" else "v4l2" if sys.platform.startswith("linux") else ""
    if kind == "usb":
        return ObsbotUSBTransport(tuning.pids, assist_off=os.environ.get("OBSBOT_ASSIST_OFF", "1") == "1",
                                  ct_unit=_env_int("OBSBOT_CT_UNIT", CT_UNIT),
                                  xu_unit=_env_int("OBSBOT_XU_UNIT", XU_UNIT),
                                  iface=_env_int("OBSBOT_IFACE", 0))
    if kind == "v4l2":  # pan/tilt/zoom only; tracking stays as OBSBOT Center left it
        return V4L2Transport(tuning.v4l2_device or f"/dev/video{source}")
    if kind == "fake":
        return FakeTransport()
    return UnavailableTransport(f"no PTZ control path for {sys.platform!r} (OBSBOT_TRANSPORT={tuning.transport})")


class ObsbotCamera(Insta360Camera):
    """CAMERAS="hero:obsbot:name:OBSBOT Tiny 2". Knobs are the Insta360 ones under
    OBSBOT_* (OBSBOT_PID, OBSBOT_HFOV_DEG, OBSBOT_TILT_SIGN, ...)."""

    backend = "obsbot"

    def __init__(self, *args, **kwargs):
        self.drift: str | None = None  # what the live position check last saw
        self._drift_prev: float | None = None
        self._drift_at = 0.0
        super().__init__(*args, **kwargs)
        self._switch_assist_off_when_streaming()

    def _tuning_from_env(self) -> Tuning:
        return Tuning.from_env("OBSBOT", DEFAULTS)

    def _make_transport(self, source: str) -> PTZTransport:
        return make_obsbot_transport(source, self.tuning)

    def _switch_assist_off_when_streaming(self) -> None:
        """Repeat the switch-off once frames flow: a camera that was asleep when the
        transport opened ignored the first one."""
        off = getattr(self.transport, "switch_assist_off", None)
        if off is None or not self._connected:
            return
        try:
            self.src.read()
            with self._lock:
                self._io(off)
        except (CameraError, TransportError) as e:
            log.warning("%s: could not repeat the tracking switch-off (%s)", self.name, e)

    def describe(self) -> dict:
        d = super().describe()
        err = getattr(self.transport, "assist_error", None)
        if "warning" in d:  # PTZ offline outranks everything below
            return d
        if self.drift:
            d.update(warning=WARN_DRIFT, tracking=f"{self.drift}: AI tracking or gesture control is on")
        elif err:
            d.update(warning=WARN_TRACKING_ON, tracking=f"may be on: {err}")
        return d

    def wide_frame(self):
        f = super().wide_frame()
        self._check_drift()
        return f

    def _check_drift(self) -> None:
        """Compare the live gimbal position with the commanded one, at most once a
        second and only well after a move settled. Reports, never corrects."""
        live_ptz = getattr(self.transport, "live_ptz", None)
        now = time.monotonic()
        if (live_ptz is None or not self._connected or self._pose_unknown or self._stuck is not None
                or now < max(self._settle_until + DRIFT_GRACE_S, self._drift_at + DRIFT_EVERY_S)):
            return
        if not self._lock.acquire(blocking=False):  # a move is being sent: not idle
            return
        try:
            self._drift_at = now
            try:
                pan_raw, tilt_raw, zoom_raw = self._io(live_ptz)
            except (TransportError, OSError):
                return  # the next move reports a dead gimbal
            tn, aim = self.tuning, self.pose
            pan = pan_raw / (tn.units_per_deg * tn.pan_sign)
            tilt = tilt_raw / (tn.units_per_deg * tn.tilt_sign)
            lo, hi = self.ranges.zoom
            zoom = 1.0 + (zoom_raw - lo) / max(hi - lo, 1) * (ZOOM_AT_MAX - 1.0)
            score = max(abs(pan - aim.pan) / DRIFT_DEG, abs(tilt - aim.tilt) / DRIFT_DEG,
                        abs(zoom - aim.zoom) / DRIFT_ZOOM)
            prev, self._drift_prev = self._drift_prev, score if score > 1 else None
            if score <= 1:
                self.drift = None
            elif prev is not None and score >= prev - 0.25:  # off twice running, not closing in
                if self.drift is None:
                    log.warning("%s: the gimbal moved on its own (live %.0f/%.0f deg %.1fx, aimed"
                                " %.0f/%.0f deg %.1fx); switch AI tracking and gesture control off"
                                " in OBSBOT Center", self.name, pan, tilt, zoom, aim.pan, aim.tilt, aim.zoom)
                self.drift = (f"moved on its own to pan {pan:.0f} tilt {tilt:.0f} zoom {zoom:.1f}x"
                              f" (aimed {aim.pan:.0f}/{aim.tilt:.0f} {aim.zoom:.1f}x)")
        finally:
            self._lock.release()

    def _zoom_top(self) -> float:
        return ZOOM_AT_MAX

    def _zoom_raw(self, factor: float) -> int:
        """1x at the raw minimum, ZOOM_AT_MAX at the raw maximum, linear in between.
        For a 100..400 range this is the Link family's factor x 100."""
        lo, hi = self.ranges.zoom
        return round(lo + (factor - 1.0) / (ZOOM_AT_MAX - 1.0) * (hi - lo))
