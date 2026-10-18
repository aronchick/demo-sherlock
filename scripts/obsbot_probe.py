# /// script
# requires-python = ">=3.11"  # link-ctl 2.2.1 needs 3.11
# dependencies = ["link-ctl>=2.2.1; sys_platform == 'darwin'", "opencv-python", "numpy"]
# ///
"""Probe an OBSBOT Tiny camera before trusting `CAMERAS=hero:obsbot:name:<name>`.

Reports: its USB vendor:product id (the Tiny SE's is not published; pass it as
OBSBOT_PID) and whether it sits behind a hub, which interface and unit ids answer
(the Tiny 2 Lite's are not published either), the pan/tilt and zoom ranges, the
vendor status block, and which OpenCV camera name to select. Quit OBSBOT Center first.

    uv run scripts/obsbot_probe.py               # read-only
    uv run scripts/obsbot_probe.py --wiggle      # switches tracking off, moves, saves snapshots/obsbot_*.jpg

--wiggle switches AI tracking and gesture control off, then checks it held: walk across
the view for 5 s and the gimbal must stay put. Then it pans +10 deg, tilts +5 deg,
zooms 2x and re-centres, timing each move from the live position. In the images +pan
must move the view RIGHT and +tilt UP; if not, set OBSBOT_PAN_SIGN=-1 /
OBSBOT_TILT_SIGN=-1. Set OBSBOT_SLEW_DEG_S from the measured speed. The snapshots show
whoever was in view: delete them once checked.
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
HUB_CLASS = 9  # USB device class of a hub


def obsbots(node: dict, hubs: tuple[str, ...] = ()) -> list[tuple[str, int, int, tuple[str, ...]]]:
    """(name, vid, pid, hubs above it) for each OBSBOT in an `ioreg -p IOUSB -a` tree."""
    out = []
    name = str(node.get("USB Product Name") or node.get("kUSBProductString") or node.get("IORegistryEntryName") or "?")
    vid, pid = node.get("idVendor"), node.get("idProduct")
    if vid is not None and (vid in VIDS or "obsbot" in name.lower()):
        out.append((name, int(vid), int(pid or 0), hubs))
    if node.get("bDeviceClass") == HUB_CLASS or (vid is not None and "hub" in name.lower()):
        hubs = (*hubs, name)
    for child in node.get("IORegistryEntryChildren", []):
        out += obsbots(child, hubs)
    return out


def usb_presence() -> None:
    print("== USB")
    r = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-a"], check=False, capture_output=True)
    try:
        tree = plistlib.loads(r.stdout) if r.returncode == 0 and r.stdout else {}
    except plistlib.InvalidFileException:
        tree = {}
    found = obsbots(tree)
    for name, vid, pid, hubs in found:
        known = pid in ob.OBSBOT_PIDS
        print(f"  {name}: {vid:04x}:{pid:04x}" + ("" if known else
              f"  <- not in the default list; run with OBSBOT_PID={pid:04x}"))
        print(f"  WARNING: behind {' > '.join(hubs)}; plug it straight into the Mac" if hubs
              else "  direct port (good)")
    if not found:
        print("  no OBSBOT on USB (vendor 3564 or 6e30). Plug it straight into the Mac.")


def answers(tr: ins.MacUSBTransport, req: int, sel: int, n: int, unit: int) -> bytes | None:
    try:
        return tr.request(req, sel, n, unit=unit)
    except ins.TransportError:
        return None


def xu_len(tr: ins.MacUSBTransport, unit: int) -> int | None:
    raw = answers(tr, ins.GET_LEN, ob.XU_SEL_SIMPLE, 2, unit)
    return struct.unpack("<H", raw)[0] if raw else None


def scan_units(pids: tuple[int, ...]) -> tuple[int, int, int | None] | None:
    """(interface, camera terminal unit, extension unit) that answer; reads only."""
    print("== Interface and unit ids (read-only scan)")
    for iface in range(4):
        tr = ob.ObsbotUSBTransport(pids, assist_off=False, iface=iface)
        try:
            tr.open()
        except ins.TransportError:
            continue
        try:
            ct = next((u for u in range(1, 9) if answers(tr, ins.GET_INFO, ins.CT_PANTILT_ABSOLUTE, 1, u)), None)
            xu = next((u for u in range(1, 9) if xu_len(tr, u) == ob.XU_LEN), None)
        finally:
            tr.close()
        if ct is not None:
            print(f"  interface {iface}: camera terminal unit {ct}, extension unit {xu}")
            return iface, ct, xu
    print("  no interface answered a pan/tilt request")
    return None


def control_path(pids: tuple[int, ...]) -> ob.ObsbotUSBTransport | None:
    want = (ob._env_int("OBSBOT_IFACE", 0), ob._env_int("OBSBOT_CT_UNIT", ins.CT_UNIT),
            ob._env_int("OBSBOT_XU_UNIT", ob.XU_UNIT))
    found = scan_units(pids)
    if found and (found[:2] != want[:2] or found[2] not in (None, want[2])):
        print(f"  differs from the defaults: export OBSBOT_IFACE={found[0]} OBSBOT_CT_UNIT={found[1]}"
              + (f" OBSBOT_XU_UNIT={found[2]}" if found[2] is not None else ""))
    iface, ct, xu = (found[0], found[1], found[2] or want[2]) if found else want
    print("== Control path: USB-direct UVC via link-ctl (IOKit), read-only")
    tr = ob.ObsbotUSBTransport(pids, assist_off=False, ct_unit=ct, xu_unit=xu, iface=iface)  # no writes here
    try:
        tr.open()
    except ins.TransportError as e:
        print(f"  NO: {e}")
        return None
    print(f"  opened {ob.OBSBOT_VID:04x}:{tr.pid:04x} interface {iface}")
    missing: list[str] = []
    for label, sel, n in (("pan/tilt", ins.CT_PANTILT_ABSOLUTE, 8), ("zoom", ins.CT_ZOOM_ABSOLUTE, 2)):
        info = answers(tr, ins.GET_INFO, sel, 1, ct)  # 0x03 = GET and SET; the Tiny 2 answers 0x03 (lkml)
        print(f"  {label:8} INFO: " + (f"0x{info[0]:02x}" if info else "unreadable"))
        for req_name, req in (("MIN", ins.GET_MIN), ("MAX", ins.GET_MAX), ("RES", ins.GET_RES),
                              ("DEF", ins.GET_DEF), ("CUR", ins.GET_CUR)):
            raw = answers(tr, req, sel, n, ct)
            if raw is None:
                print(f"  {label:8} {req_name}: unreadable")
                if req in (ins.GET_MIN, ins.GET_MAX):
                    missing.append(f"{label} {req_name}")
            elif n == 8:
                pan, tilt = struct.unpack("<ii", raw)
                print(f"  {label:8} {req_name}: pan={pan} tilt={tilt} raw"
                      f"  (= {pan / 3600:.1f} / {tilt / 3600:.1f} deg if arc-seconds)")
            else:
                print(f"  {label:8} {req_name}: {struct.unpack('<H', raw)[0]}")
    status = read_status(tr)
    if status is None:
        print(f"  extension unit {xu}: unreadable; tracking cannot be switched off from here,"
              " use OBSBOT Center")
    else:
        print(f"  extension unit {xu} status: {status.hex(' ')}")
        if tr.pid == 0xFEF8 and len(status) > 0x1C:  # cgevans/tiny2's layout; the Lite's is undocumented
            print(f"  AI mode (Tiny 2 layout, bytes 0x18/0x1c): {status[0x18]}/{status[0x1C]}"
                  + ("  <- tracking ON" if status[0x18] else ""))
    if missing:
        print(f"  NO: the camera opened but did not answer {', '.join(missing)}")
        tr.close()
        return None
    return tr


def read_status(tr: ob.ObsbotUSBTransport) -> bytes | None:
    length = xu_len(tr, tr.xu_unit)
    return answers(tr, ins.GET_CUR, ob.XU_SEL_SIMPLE, length, tr.xu_unit) if length else None


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
    before = read_status(tr)
    tr.switch_assist_off()
    print(f"  {tr.assist_error or 'AI tracking and gesture control switched off'}")
    tuning = ins.Tuning.from_env("OBSBOT", ob.DEFAULTS)
    tuning.home_after_s = 0
    # One USB handle: the camera drives tr while the probe reads the live position from it.
    # Opening it homes the gimbal and repeats the switch-off once frames flow.
    cam = ob.ObsbotCamera("hero", f"name:{name}", transport=tr, tuning=tuning)
    after = read_status(tr)
    if before and after:  # which status bytes the switch-off changed: maps the Lite's layout
        changed = {f"0x{i:02x}": f"{b}->{a}" for i, (b, a) in enumerate(zip(before, after, strict=False)) if a != b}
        print(f"  status bytes changed by the switch-off: {changed or 'none'}")
    print("  ", cam.describe())
    out = ROOT / "snapshots"
    out.mkdir(exist_ok=True)
    try:
        settle(cam, time.monotonic())
        hold_still(tr)
        for label, kw in (("home", {"pan": 0, "tilt": 0, "zoom": 1}), ("pan+10", {"pan": 10}),
                          ("tilt+5", {"pan": 0, "tilt": 5}), ("zoom2x", {"tilt": 0, "zoom": 2}),
                          ("home-again", {"pan": 0, "tilt": 0, "zoom": 1})):
            t0 = time.monotonic()
            start = live_pose(tr)
            pose = cam.move(**kw)
            moved = settle(cam, t0)
            path = out / f"obsbot_{label}.jpg"
            cv2.imwrite(str(path), cam.snapshot())
            print(f"  {label:10} -> {pose}  live {start} -> {moved[0]} in {moved[1]:.2f}s"
                  f"  {path.relative_to(ROOT)}")
    finally:
        try:
            cam.move(pan=0, tilt=0, zoom=1)
            settle(cam, time.monotonic())
        finally:
            cam.close()
    print("  delete snapshots/obsbot_*.jpg once checked: they show whoever was in view")


def hold_still(tr: ob.ObsbotUSBTransport, seconds: float = 5.0) -> None:
    print(f"  hold-still test: walk across the camera's view now, for {seconds:.0f}s")
    start = live_pose(tr)
    worst = 0.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        now = live_pose(tr)
        worst = max(worst, abs(now[0] - start[0]), abs(now[1] - start[1]))
        time.sleep(0.2)
    print(f"  TRACKING STILL ON: the gimbal followed by {worst:.1f} deg. Switch AI tracking and"
          " gesture control off in OBSBOT Center, quit it, and rerun" if worst > ob.DRIFT_DEG
          else f"  held still (moved at most {worst:.1f} deg)")


def live_pose(tr: ins.MacUSBTransport) -> tuple[float, float]:
    pan, tilt = struct.unpack("<ii", tr.request(ins.GET_CUR, ins.CT_PANTILT_ABSOLUTE, 8))
    return round(pan / 3600, 1), round(tilt / 3600, 1)


def settle(cam: ob.ObsbotCamera, t0: float, timeout: float = 10.0) -> tuple[tuple[float, float], float]:
    """Require pan, tilt AND zoom to reach the target, not merely stop changing."""
    tn, pose = cam.tuning, cam.pose
    target = (round(pose.pan * tn.units_per_deg * tn.pan_sign),
              round(pose.tilt * tn.units_per_deg * tn.tilt_sign), cam._zoom_raw(pose.zoom))
    matched = 0
    while time.monotonic() - t0 < timeout:
        raw = cam.transport.live_ptz()
        at_target = all(abs(a - b) <= tol for a, b, tol in zip(raw, target, (3600, 3600, 1), strict=True))
        matched = matched + 1 if at_target else 0
        if matched >= 3:
            return (raw[0] / 3600, raw[1] / 3600), time.monotonic() - t0
        time.sleep(0.1)
    raise RuntimeError(f"PTZ did not reach {target} within {timeout}s; last readback {raw}")


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
