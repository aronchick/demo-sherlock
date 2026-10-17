# /// script
# requires-python = ">=3.10"
# dependencies = ["link-ctl>=2.2.1; sys_platform == 'darwin'", "opencv-python", "numpy"]
# ///
"""Probe the Insta360 Link 2 hero camera before trusting `CAMERAS=hero:insta360:<index>`.

Reports: is it on USB (and behind a hub?), which control path answers, the PTZ
ranges it reports, whether AI tracking is on (it would fight our moves), and
which OpenCV index is its video stream.

    uv run scripts/ptz_probe.py                  # read-only
    uv run scripts/ptz_probe.py --wiggle         # moves the gimbal, saves snapshots/probe_*.jpg
    uv run scripts/ptz_probe.py --wiggle --index 1

--wiggle pans +10 deg, tilts +5 deg, zooms 2x, then re-centres. In the saved
images +pan must move the view RIGHT and +tilt UP; if not, set
INSTA360_PAN_SIGN=-1 / INSTA360_TILT_SIGN=-1.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sherlock"))  # sherlock/ is not an installed package
ins = importlib.import_module("sherlock.camera.insta360")


AI_MODES = {0x00: "normal", 0x01: "TRACKING", 0x04: "whiteboard", 0x05: "overhead",
            0x06: "deskview", 0xFF: "switching"}


def usb_presence() -> None:
    print("== USB")
    if sys.platform != "darwin":
        if shutil.which("lsusb"):
            out = subprocess.run(["lsusb", "-d", f"{ins.INSTA360_VID:04x}:"], check=False,
                                 capture_output=True, text=True).stdout.strip()
            print(out or "  no Insta360 device on USB")
        return
    found = False
    for dtype in ("SPUSBHostDataType", "SPUSBDataType"):
        r = subprocess.run(["system_profiler", "-json", dtype], check=False, capture_output=True,
                           text=True)
        try:
            tree = json.loads(r.stdout or "{}").get(dtype, [])
        except json.JSONDecodeError:
            continue

        def walk(items: list, depth: int, parents: list[str]) -> None:
            nonlocal found
            for it in items:
                flat = {k: v for k, v in it.items() if k != "_items"}
                text = json.dumps(flat).lower()
                if "insta360" in text or f"0x{ins.INSTA360_VID:04x}" in text:
                    found = True
                    hub = [p for p in parents[1:] if "hub" in p.lower()]
                    print(f"  found: {flat}")
                    print(f"  path: {' > '.join(parents)}")
                    print("  WARNING: behind a hub; plug it straight into the Mac" if hub
                          else "  direct port (good: Link 2 is USB 2.0, keep it off the hub)")
                walk(it.get("_items", []), depth + 1, [*parents, str(it.get("_name", "?"))])

        walk(tree, 0, [])
        if found:
            return
    print("  no Insta360 device on USB. Plug the Link 2 straight into the Mac (not the hub).")


def macos_usb_direct() -> ins.MacUSBTransport | None:
    print("== Control path 1: USB-direct UVC via link-ctl (IOKit)")
    tr = ins.MacUSBTransport()
    try:
        tr.open()
    except ins.TransportError as e:
        print(f"  NO: {e}")
        return None
    print(f"  opened {ins.INSTA360_VID:04x}:{tr.pid:04x} video-control interface")
    for label, sel, n in (("pan/tilt", ins.CT_PANTILT_ABSOLUTE, 8), ("zoom", ins.CT_ZOOM_ABSOLUTE, 2)):
        for req_name, req in (("MIN", ins.GET_MIN), ("MAX", ins.GET_MAX), ("RES", ins.GET_RES),
                              ("DEF", ins.GET_DEF), ("CUR", ins.GET_CUR)):
            try:
                raw = tr.request(req, sel, n)
            except ins.TransportError as e:
                print(f"  {label:8} {req_name}: unreadable ({e})")
                continue
            if n == 8:
                pan, tilt = struct.unpack("<ii", raw)
                print(f"  {label:8} {req_name}: pan={pan} tilt={tilt} raw"
                      f"  (= {pan / 3600:.1f} / {tilt / 3600:.1f} deg if arc-seconds)")
            else:
                print(f"  {label:8} {req_name}: {struct.unpack('<H', raw)[0]}")
    try:  # AI mode lives in Insta360 extension unit 9, selector 0x02 (61 bytes on Link 2)
        (length,) = struct.unpack("<H", tr.request(ins.GET_LEN, 0x02, 2, unit=9))
        mode = tr.request(ins.GET_CUR, 0x02, length, unit=9)[0]
        name = AI_MODES.get(mode, f"0x{mode:02x}")
        print(f"  AI mode: {name}" + ("  <- turn tracking OFF in Link Controller, it will fight"
                                      " look_at" if mode == 0x01 else ""))
    except ins.TransportError as e:
        print(f"  AI mode: unreadable ({e})")
    return tr


def link_controller_app() -> None:
    print("== Control path 2: Link Controller WebSocket")
    r = subprocess.run(["pgrep", "-fl", "Insta360 Link Controller"], check=False,
                       capture_output=True, text=True)
    print("  app running" if r.stdout.strip() else "  app not running")
    print("  not used: the WS protocol has zoom, re-centre and joystick velocity, but no"
          " absolute pan/tilt, so it cannot aim at a bbox")


def linux_v4l2(device: str) -> ins.V4L2Transport | None:
    print(f"== Control path 3: v4l2 on {device}")
    tr = ins.V4L2Transport(device)
    try:
        tr.open()
    except ins.TransportError as e:
        print(f"  NO: {e}")
        return None
    for name in ("pan_absolute", "tilt_absolute", "zoom_absolute"):
        print(f"  {name}: {tr.ctrls.get(name, 'missing')}")
    return tr


def video_indexes() -> int | None:
    print("== Video streams (OpenCV index -> size)")
    names: dict[int, str] = {}
    if sys.platform == "darwin" and shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices",
                            "true", "-i", ""], check=False, capture_output=True, text=True)
        for ln in r.stderr.splitlines():
            if "audio devices" in ln:
                break
            m = re.search(r"\] \[(\d+)\] (.+)$", ln)
            if m:
                names[int(m.group(1))] = m.group(2)
    guess = next((i for i, n in names.items() if "insta360" in n.lower()), None)
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            print(f"  index {i}: {w}x{h}  {names.get(i, '')}")
    if guess is not None:
        print(f"  Link 2 looks like index {guess}:"
              f" CAMERAS=hero:insta360:{guess},scout1:fake:<mac webcam index>")
    else:
        print("  could not name the Link 2 stream (no ffmpeg, or not listed); pick by size"
              " or pass --index")
    return guess


def wiggle(index: int) -> None:
    print(f"== Wiggle test on index {index}")
    tuning = ins.Tuning.from_env()
    tuning.home_after_s = 0
    cam = ins.Insta360Camera("hero", str(index), tuning=tuning)
    print("  ", cam.describe())
    out = ROOT / "snapshots"
    out.mkdir(exist_ok=True)
    try:
        for label, kw in (("home", {"pan": 0, "tilt": 0, "zoom": 1}), ("pan+10", {"pan": 10}),
                          ("tilt+5", {"pan": 0, "tilt": 5}), ("zoom2x", {"tilt": 0, "zoom": 2}),
                          ("home-again", {"pan": 0, "tilt": 0, "zoom": 1})):
            t0 = time.monotonic()
            pose = cam.move(**kw)
            img = cam.snapshot()
            path = out / f"probe_{label}.jpg"
            cv2.imwrite(str(path), img)
            print(f"  {label:10} -> {pose}  settled+captured in {time.monotonic() - t0:.2f}s"
                  f"  {path.relative_to(ROOT)}")
    finally:
        cam.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wiggle", action="store_true", help="move the gimbal and save snapshots")
    ap.add_argument("--index", type=int, help="OpenCV index of the Link 2 stream")
    ap.add_argument("--device", default="/dev/video0", help="v4l2 device (Linux)")
    a = ap.parse_args()

    usb_presence()
    if sys.platform == "darwin":
        tr = macos_usb_direct()
        link_controller_app()
    else:
        tr = linux_v4l2(a.device)
    guess = video_indexes()
    print("== Verdict")
    if tr is None:
        print("  no PTZ control path answered; the Director will get CameraError for hero"
              " and fall back to the next camera")
        return 1
    print(f"  PTZ control works via {tr.name}")
    tr.close()
    index = a.index if a.index is not None else guess
    if a.wiggle:
        if index is None:
            print("  --wiggle needs --index (no stream found)")
            return 1
        wiggle(index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
