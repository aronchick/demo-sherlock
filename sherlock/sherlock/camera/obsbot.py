"""OBSBOT Tiny 2 / Tiny 2 Lite adapter.

Plan A: standard UVC Camera Terminal controls (CT_PANTILT_ABSOLUTE 0x0D,
CT_ZOOM_ABSOLUTE 0x0B). Verified on the Tiny 2; test on the Lite first thing.
On macOS the simplest route is the `uvc-util` CLI or `pyuvc`; on Linux use
v4l2-ctl --set-ctrl=pan_absolute / tilt_absolute / zoom_absolute.

Plan B: OSC through OBSBOT Center if the Lite ignores raw UVC.

Only the `move` method differs from FakePTZCamera; the Scout still gets a
wide frame by moving to home+zoom 1, so this subclasses it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from .base import CameraError, FakePTZCamera, Pose

# UVC pan/tilt units are arc-seconds (1 degree = 3600).
_ARCSEC = 3600


class ObsbotCamera(FakePTZCamera):
    def __init__(self, name: str, source: str, device: str | None = None):
        super().__init__(name, source)
        self.device = device or ("/dev/video" + source if sys.platform.startswith("linux") else source)
        self._driver = self._pick_driver()

    def _pick_driver(self) -> str:
        if sys.platform.startswith("linux") and shutil.which("v4l2-ctl"):
            return "v4l2"
        if shutil.which("uvc-util"):
            return "uvc-util"
        return "software"  # fall back to cropping; the demo still runs

    def move(self, pan=None, tilt=None, zoom=None) -> Pose:
        pose = super().move(pan, tilt, zoom)  # clamps + keeps software crop in sync
        if self._driver == "software":
            return pose
        try:
            if self._driver == "v4l2":
                ctrls = f"pan_absolute={int(pose.pan * _ARCSEC)},tilt_absolute={int(pose.tilt * _ARCSEC)}," \
                        f"zoom_absolute={int(pose.zoom * 100)}"
                subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl", ctrls],
                               check=True, capture_output=True, timeout=2)
            else:  # uvc-util on macOS: one call per control
                for ctl, val in (("pan-tilt-abs", f"{int(pose.pan * _ARCSEC)},{int(pose.tilt * _ARCSEC)}"),
                                 ("zoom-abs", str(int(pose.zoom * 100)))):
                    subprocess.run(["uvc-util", "-I", "0", "-s", f"{ctl}={val}"],
                                   check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise CameraError(f"{self.name}: PTZ command failed: {e}") from e
        return pose

    def snapshot(self):
        # With real optics the stream already is the zoomed view; don't double-crop.
        if self._driver != "software":
            self._check()
            return self.src.read()
        return super().snapshot()
