from .base import Camera, CameraError, FakePTZCamera, Pose
from .obsbot import ObsbotCamera


def build_camera(name: str, backend: str, source: str) -> Camera:
    if backend == "obsbot":
        return ObsbotCamera(name, source)
    return FakePTZCamera(name, None if source == "synthetic" else source)
