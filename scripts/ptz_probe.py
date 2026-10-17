"""Does this camera answer standard UVC pan/tilt/zoom? Run before trusting ObsbotCamera.
macOS: brew install uvc-util (or pip install pyuvc). Linux: v4l2-ctl from v4l-utils."""
import shutil
import subprocess
import sys

if sys.platform.startswith("linux"):
    if not shutil.which("v4l2-ctl"):
        sys.exit("install v4l-utils")
    print(subprocess.run(["v4l2-ctl", "-d", sys.argv[1] if len(sys.argv) > 1 else "/dev/video0", "-L"],
                         check=False, capture_output=True, text=True).stdout)
    print("Look for pan_absolute / tilt_absolute / zoom_absolute above. If present, Plan A works.")
else:
    if shutil.which("uvc-util"):
        print(subprocess.run(["uvc-util", "-I", "0", "-c"], check=False, capture_output=True, text=True).stdout)
        print("Look for pan-tilt-abs / zoom-abs. If present, Plan A works.")
    else:
        try:
            import uvc  # pyuvc
            dev = uvc.device_list()
            print(dev)
            cap = uvc.Capture(dev[0]["uid"])
            for c in cap.controls:
                print(c.display_name, c.value, c.min_val, c.max_val)
            print("Look for Pan / Tilt / Zoom controls. If present, Plan A works.")
        except Exception as e:  # noqa
            sys.exit(f"no probe tool available ({e}); brew install uvc-util or pip install pyuvc")
