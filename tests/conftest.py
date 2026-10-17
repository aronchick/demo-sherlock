"""Mock-mode environment shared by every test. Set before any sherlock import,
because Config reads the environment when it is constructed."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sherlock"))

os.environ["SHERLOCK_MOCK"] = "1"
os.environ["CAMERAS"] = "hero:fake:synthetic,scout1:fake:synthetic"
os.environ["DWELL_SECONDS"] = "0.3"
os.environ["SCENE_SPEED"] = "3"  # cup at ~1 s, knife at ~5 s, loop every ~9 s
os.environ["MOCK_LATENCY"] = "0.1"
