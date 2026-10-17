from .base import Camera, CameraError, FakePTZCamera, Pose
from .insta360 import Insta360Camera
from .obsbot import ObsbotCamera

__all__ = ["Camera", "CameraError", "FakePTZCamera", "Insta360Camera", "ObsbotCamera", "Pose", "build_camera"]


def build_camera(name: str, backend: str, source: str) -> Camera:
    if backend == "insta360":
        return Insta360Camera(name, source)
    if backend == "obsbot":
        return ObsbotCamera(name, source)
    return FakePTZCamera(name, None if source == "synthetic" else source)
