"""OBSBOT Tiny 2 / Tiny 2 Lite / Tiny SE adapter. DRAFT: not yet run on a real OBSBOT.

The OBSBOT Tiny family is a UVC camera on a two-axis gimbal, so this is the Insta360
adapter (same IOKit control transfers through link-ctl, same aiming, settling and
fault handling) with OBSBOT's USB ids, specs and two differences:

- AI tracking pans, tilts and zooms on its own and would fight every look_at, so
  open() switches it off through OBSBOT's vendor extension unit.
- Zoom is mapped over whatever raw range the camera reports, instead of the Link
  family's raw = factor x 100.

Facts, with sources (checked 2026-10-17):
- USB ids: vendor 0x3564 (Remo Tech). Tiny 2 = fef8: Linux kernel patch thread
  "uvcvideo: fix up missing AUTO_UPDATE on OBSBOT Tiny 2 pan/tilt" (lkml, 2026-08/09,
  full lsusb -v), plus three Linux control projects. Tiny 2 Lite = fef9: lsusb on the
  openSUSE forum and Reddit, and the device tables of Domatix/obsbot-control and
  samuelcolvin/obsbot-rs. Tiny SE: no public source found; read it with
  scripts/obsbot_probe.py and pass OBSBOT_PID=<hex>.
- Camera Terminal is unit 1 and advertises PanTilt (Absolute) and Zoom (Absolute)
  (the lkml lsusb -v), so pan and tilt are arc-seconds per the UVC spec. Device
  GET_CUR reports the live gimbal position.
- Tracking off: extension unit 2, selector 6, 60-byte payload [0x16, 2, 0, 0, 0...]
  (AI mode 0, sub-mode 0). Two independent implementations agree: cgevans/tiny2
  (EUPL-1.2, Tiny 2) and samuelcolvin/obsbot-rs (MIT, Tiny 2 Lite). Not from OBSBOT;
  the official route is OBSBOT Center or its SDK (sign-up required).
- Specs (obsbot.com tech specs): pan +-140 deg, tilt 30 deg down to 70 deg up,
  120 deg/s, zoom 1-4x; horizontal FOV 72.9 deg (Tiny 2), 71 deg (Tiny SE).
- The camera sleeps after about a minute without a stream and then ignores writes
  (obsbot-rs); Sherlock streams continuously, so the gimbal should stay awake.

Without hardware control (no link-ctl, Linux, or a model that refuses), use the
software crop on the same stream instead: CAMERAS="hero:fake:name:OBSBOT Tiny 2".
"""
from __future__ import annotations

import logging
import os
import sys

from .insta360 import (
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
TAG_AI_MODE = 0x16

DEFAULTS = Tuning(pids=OBSBOT_PIDS, hfov_deg=72.9, max_zoom=4.0, slew_deg_s=120.0)
# Zoom factor at the camera's raw maximum: the spec sheet's 4x.
ZOOM_AT_MAX = 4.0


def tracking_payload(on: bool) -> bytes:
    return bytes([TAG_AI_MODE, 2, 2 if on else 0, 0]).ljust(XU_LEN, b"\0")


class ObsbotUSBTransport(MacUSBTransport):
    """link-ctl's IOKit control transfers, aimed at an OBSBOT. Switches AI tracking off
    on every fresh open, since a replugged camera comes back in its boot mode."""

    name = "obsbot-usb"

    def __init__(self, pids: tuple[int, ...] = OBSBOT_PIDS, tracking_off: bool = True):
        super().__init__(pids, vid=OBSBOT_VID, label="OBSBOT camera")
        self.tracking_off = tracking_off
        self.tracking_error: str | None = None

    def open(self) -> None:
        fresh = self._h is None
        super().open()
        if fresh and self.tracking_off:
            try:
                self.set_tracking(False)
                self.tracking_error = None
            except TransportError as e:
                # Pan, tilt and zoom are standard UVC and still work; tracking may fight them.
                self.tracking_error = str(e)
                log.warning("OBSBOT: could not switch AI tracking off (%s); turn it off in"
                            " OBSBOT Center", e)

    def set_tracking(self, on: bool) -> None:
        self._set(XU_SEL_SIMPLE, tracking_payload(on), unit=XU_UNIT)


def make_obsbot_transport(source: str, tuning: Tuning) -> PTZTransport:
    kind = tuning.transport
    if kind == "auto":
        kind = "usb" if sys.platform == "darwin" else "v4l2" if sys.platform.startswith("linux") else ""
    if kind == "usb":
        return ObsbotUSBTransport(tuning.pids, tracking_off=os.environ.get("OBSBOT_TRACKING_OFF", "1") == "1")
    if kind == "v4l2":  # pan/tilt/zoom only; tracking stays as OBSBOT Center left it
        return V4L2Transport(tuning.v4l2_device or f"/dev/video{source}")
    if kind == "fake":
        return FakeTransport()
    return UnavailableTransport(f"no PTZ control path for {sys.platform!r} (OBSBOT_TRANSPORT={tuning.transport})")


class ObsbotCamera(Insta360Camera):
    """CAMERAS="hero:obsbot:name:OBSBOT Tiny 2". Knobs are the Insta360 ones under
    OBSBOT_* (OBSBOT_PID, OBSBOT_HFOV_DEG, OBSBOT_TILT_SIGN, ...)."""

    backend = "obsbot"

    def _tuning_from_env(self) -> Tuning:
        return Tuning.from_env("OBSBOT", DEFAULTS)

    def _make_transport(self, source: str) -> PTZTransport:
        return make_obsbot_transport(source, self.tuning)

    def describe(self) -> dict:
        d = super().describe()
        err = getattr(self.transport, "tracking_error", None)
        if err:
            d["tracking"] = f"may be on: {err}"
        return d

    def _zoom_top(self) -> float:
        return ZOOM_AT_MAX

    def _zoom_raw(self, factor: float) -> int:
        """1x at the raw minimum, ZOOM_AT_MAX at the raw maximum, linear in between.
        For a 100..400 range this is the Link family's factor x 100."""
        lo, hi = self.ranges.zoom
        return round(lo + (factor - 1.0) / (ZOOM_AT_MAX - 1.0) * (hi - lo))
