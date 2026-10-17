# /// script
# requires-python = ">=3.11"  # link-ctl 2.2.1 needs 3.11
# dependencies = ["link-ctl>=2.2.1; sys_platform == 'darwin'", "opencv-python", "numpy"]
# ///
"""Probe an OBSBOT Tiny camera before trusting `CAMERAS=hero:obsbot:name:<name>`.

Reports: its USB vendor:product id (the Tiny SE's is not published; pass it as
OBSBOT_PID), whether the IOKit control path answers, the Camera Terminal pan/tilt and
zoom ranges, the vendor extension unit and AI tracking state, and which OpenCV
camera name to select.

    uv run scripts/obsbot_probe.py               # read-only
    uv run scripts/obsbot_probe.py --wiggle      # tracking off, moves, saves snapshots/obsbot_*.jpg

--wiggle pans +10 deg, tilts +5 deg, zooms 2x, then re-centres, timing each move from
the live position. In the images +pan must move the view RIGHT and +tilt UP; if not,
set OBSBOT_PAN_SIGN=-1 / OBSBOT_TILT_SIGN=-1. Set OBSBOT_SLEW_DEG_S from the measured speed.
"""
from __future__ import annotations

import argparse
import importlib
import plistlib
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sherlock"))  # sherlock/ is not an installed package
ins = importlib.import_module("sherlock.camera.insta360")
ob = importlib.import_module("sherlock.camera.obsbot")
base = importlib.import_module("sherlock.camera.base")

VIDS = (ob.OBSBOT_VID, 0x6E30)  # 0x6e30: first-generation Tiny and Meet 4K (obsbot-rs)


def obsbots(devices: list[dict]) -> list[tuple[str, int, int]]:
    """(name, vid, pid) for each OBSBOT in an `ioreg -a` IOUSBHostDevice list."""
    out = []
    for d in devices:
        vid, pid = d.get("idVendor"), d.get("idProduct")
        name = d.get("USB Product Name") or d.get("kUSBProductString") or "?"
        # The name check also catches a key renamed in some macOS release.
        if vid in VIDS or "obsbot" in str(d).lower():
            out.append((str(name), int(vid or 0), int(pid or 0)))
    return out


def usb_presence() -> list[tuple[str, int, int]]:
    print("== USB")
    r = subprocess.run(["ioreg", "-a", "-r", "-c", "IOUSBHostDevice", "-l"], check=False,
                       capture_output=True)
    try:
        devices = plistlib.loads(r.stdout) if r.returncode == 0 and r.stdout else []
    except plistlib.InvalidFileException:
        devices = []
    found = obsbots(devices)
    for name, vid, pid in found:
        known = pid in ob.OBSBOT_PIDS
        print(f"  {name}: {vid:04x}:{pid:04x}" + ("" if known else
              f"  <- not in the default list; run with OBSBOT_PID={pid:04x}"))
    if not found:
        print("  no OBSBOT on USB (vendor 3564 or 6e30). Plug it straight into the Mac.")
    return found


def control_path(pids: tuple[int, ...]) -> ob.ObsbotUSBTransport | None:
    print("== Control path: USB-direct UVC via link-ctl (IOKit), read-only")
    tr = ob.ObsbotUSBTransport(pids, tracking_off=False)  # no writes in this step
    try:
        tr.open()
    except ins.TransportError as e:
        print(f"  NO: {e}")
        return None
    print(f"  opened {ob.OBSBOT_VID:04x}:{tr.pid:04x} video-control interface")
    missing: list[str] = []
    for label, sel, n in (("pan/tilt", ins.CT_PANTILT_ABSOLUTE, 8), ("zoom", ins.CT_ZOOM_ABSOLUTE, 2)):
        try:  # 0x03 = GET and SET; the Tiny 2 answers 0x03 (lkml)
            print(f"  {label:8} INFO: 0x{tr.request(ins.GET_INFO, sel, 1)[0]:02x}")
        except ins.TransportError as e:
            print(f"  {label:8} INFO: unreadable ({e})")
        for req_name, req in (("MIN", ins.GET_MIN), ("MAX", ins.GET_MAX), ("RES", ins.GET_RES),
                              ("DEF", ins.GET_DEF), ("CUR", ins.GET_CUR)):
            try:
                raw = tr.request(req, sel, n)
            except ins.TransportError as e:
                print(f"  {label:8} {req_name}: unreadable ({e})")
                if req in (ins.GET_MIN, ins.GET_MAX):
                    missing.append(f"{label} {req_name}")
                continue
            if n == 8:
                pan, tilt = struct.unpack("<ii", raw)
                print(f"  {label:8} {req_name}: pan={pan} tilt={tilt} raw"
                      f"  (= {pan / 3600:.1f} / {tilt / 3600:.1f} deg if arc-seconds)")
            else:
                print(f"  {label:8} {req_name}: {struct.unpack('<H', raw)[0]}")
    try:  # vendor extension unit 2, selector 6: a 60-byte status block
        (length,) = struct.unpack("<H", tr.request(ins.GET_LEN, ob.XU_SEL_SIMPLE, 2, unit=ob.XU_UNIT))
        status = tr.request(ins.GET_CUR, ob.XU_SEL_SIMPLE, length, unit=ob.XU_UNIT)
        # Bytes 0x18 and 0x1c hold the AI mode and sub-mode (cgevans/tiny2); 0 = no tracking.
        mode = (status[0x18], status[0x1C]) if length > 0x1C else None
        print(f"  extension unit 2 selector 6: {length} bytes (expect 60); AI mode {mode}"
              + ("  <- tracking ON: it will fight look_at" if mode and mode[0] else ""))
    except ins.TransportError as e:
        print(f"  extension unit 2: unreadable ({e}); tracking-off will not work, use OBSBOT Center")
    if missing:
        print(f"  NO: the camera opened but did not answer {', '.join(missing)}")
        tr.close()
        return None
    return tr


def video_names() -> str | None:
    print("== Video streams (OpenCV index -> name, size)")
    try:
        devices = base.list_video_devices()  # same indexes OpenCV uses
    except base.CameraError as e:
        print(f"  cannot list camera names ({e})")
        return None
    for i, name in devices:
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        size = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no frame (camera permission?)"
        print(f"  index {i}: {name:28} {size}")
    names = [n for _, n in devices if "obsbot" in n.lower() and "virtual" not in n.lower()]
    if any("virtual" in n.lower() for _, n in devices):
        print("  OBSBOT Center's virtual camera is listed: select the real camera by its full name")
    if len(names) == 1:
        print(f'  CAMERAS="hero:obsbot:name:{names[0]},scout1:fake:name:MacBook Pro Camera"')
        return names[0]
    print("  no OBSBOT stream in the list" if not names else f"  several OBSBOT streams: {names}")
    return None


def wiggle(name: str, tr: ob.ObsbotUSBTransport) -> None:
    print(f"== Wiggle test on {name!r}")
    try:
        tr.set_tracking(False)
        print("  AI tracking switched off")
    except ins.TransportError as e:
        print(f"  could not switch AI tracking off ({e}); moves may be undone by tracking")
    tuning = ins.Tuning.from_env("OBSBOT", ob.DEFAULTS)
    tuning.home_after_s = 0
    # One USB handle: the camera drives tr while the probe reads the live position from it.
    cam = ob.ObsbotCamera("hero", f"name:{name}", transport=tr, tuning=tuning)
    print("  ", cam.describe())
    out = ROOT / "snapshots"
    out.mkdir(exist_ok=True)
    try:
        for label, kw in (("home", {"pan": 0, "tilt": 0, "zoom": 1}), ("pan+10", {"pan": 10}),
                          ("tilt+5", {"pan": 0, "tilt": 5}), ("zoom2x", {"tilt": 0, "zoom": 2}),
                          ("home-again", {"pan": 0, "tilt": 0, "zoom": 1})):
            t0 = time.monotonic()
            before = live_pose(tr)
            pose = cam.move(**kw)
            moved = settle(tr, t0)
            img = cam.snapshot()
            path = out / f"obsbot_{label}.jpg"
            cv2.imwrite(str(path), img)
            print(f"  {label:10} -> {pose}  live {before} -> {moved[0]} in {moved[1]:.2f}s"
                  f"  {path.relative_to(ROOT)}")
    finally:
        cam.close()


def live_pose(tr: ins.MacUSBTransport) -> tuple[float, float]:
    pan, tilt = struct.unpack("<ii", tr.request(ins.GET_CUR, ins.CT_PANTILT_ABSOLUTE, 8))
    return round(pan / 3600, 1), round(tilt / 3600, 1)


def settle(tr: ins.MacUSBTransport, t0: float, timeout: float = 5.0) -> tuple[tuple[float, float], float]:
    """Poll the live position until two reads 100 ms apart agree: when the gimbal stopped."""
    last = live_pose(tr)
    while time.monotonic() - t0 < timeout:
        time.sleep(0.1)
        now = live_pose(tr)
        if now == last:
            return now, time.monotonic() - t0
        last = now
    return last, timeout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wiggle", action="store_true", help="switch tracking off, move, save snapshots")
    a = ap.parse_args()
    if sys.platform != "darwin":
        print("macOS only: on Linux use v4l2-ctl --list-ctrls on the OBSBOT's /dev/videoN")
        return 1
    usb_presence()
    tuning = ins.Tuning.from_env("OBSBOT", ob.DEFAULTS)
    tr = control_path(tuning.pids)
    name = video_names()
    print("== Verdict")
    if tr is None:
        print("  no PTZ control path answered. Fallback: software crop on the same stream,"
              f' CAMERAS="hero:fake:name:{name or "<OBSBOT name>"}"')
        return 1
    print(f"  PTZ control works via {tr.name}")
    try:
        if a.wiggle:
            if name is None:
                print("  --wiggle needs exactly one OBSBOT stream")
                return 1
            wiggle(name, tr)
    finally:
        tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
