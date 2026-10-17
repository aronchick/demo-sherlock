"""Only the masked object crop reaches a model, and only the Verifier decides. The real Analyst and
Verifier clients talk to a loopback OpenAI-compatible stub that records every request, so these
tests read the exact image bytes each model received."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from sherlock.agents.vision import Analyst, Verifier, _b64
from sherlock.config import Endpoint
from test_weapon_flow import _frozen_sherlock, _knife_det

YES = {"content": [{"token": "Yes", "logprob": -0.05, "bytes": None,
                    "top_logprobs": [{"token": "Yes", "logprob": -0.05, "bytes": None},
                                     {"token": "No", "logprob": -3.05, "bytes": None}]},
                   {"token": ".", "logprob": 0.0, "bytes": None, "top_logprobs": []}], "refusal": None}


@pytest.fixture
def endpoint():
    bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            verifier = body["messages"][0]["content"][1]["text"].endswith("Answer yes or no.")
            data = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "logprobs": YES if verifier else None,
                                            "message": {"role": "assistant",
                                                        "content": "Yes." if verifier else "A kitchen knife."}}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield Endpoint(f"http://127.0.0.1:{server.server_address[1]}/v1", "vlm", "unused"), bodies
    server.shutdown()
    server.server_close()


def _images(bodies):
    return [part["image_url"]["url"] for body in bodies for part in body["messages"][0]["content"]
            if part["type"] == "image_url"]


def test_no_face_or_full_frame_bytes_reach_any_model(tmp_path, endpoint):
    ep, bodies = endpoint
    s = _frozen_sherlock(tmp_path)
    s.tools.analyst, s.tools.verifier = Analyst(ep), Verifier(ep)
    s._investigate(_knife_det())
    report, evidence = s.log[-1], s.tools.evidence
    # The face crop and the full frame are kept for a human reviewer ...
    assert {"original_frame", "face_crop", "weapon_crop"} <= set(report["evidence"])
    # ... and every model request carries the masked weapon crop, byte for byte, and nothing else.
    images = _images(bodies)
    assert len(bodies) == len(images) == 2
    assert set(images) == {f"data:image/jpeg;base64,{_b64(evidence['weapon_crop'])}"}
    for name in ("face_crop", "original_frame"):
        assert not any(_b64(evidence[name]) in image for image in images), f"{name} reached a model"
    for image in images:
        sent = cv2.imdecode(np.frombuffer(base64.b64decode(image.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        assert sent.shape == evidence["weapon_crop"].shape
    assert "model_context" not in evidence and "model_context" not in report["evidence"]
    assert report["action"] == "alert" and report["comparison"]["status"] == "confirmed"


def test_model_prompts_ask_about_the_object_only(endpoint, tmp_path):
    ep, bodies = endpoint
    s = _frozen_sherlock(tmp_path)
    s.tools.analyst, s.tools.verifier = Analyst(ep), Verifier(ep)
    s._investigate(_knife_det())
    prompts = [body["messages"][0]["content"][1]["text"] for body in bodies]
    assert any(p.startswith("Is the object in this image a knife?") for p in prompts)
    for prompt in prompts:
        assert "scene" not in prompt and "person context" not in prompt and "JSON" not in prompt


@pytest.mark.parametrize("said", ["A kitchen knife.", "Cake.", "", None])  # None: the Analyst fails
@pytest.mark.parametrize("yes", [True, False])
def test_the_verifier_alone_decides(tmp_path, said, yes):
    s = _frozen_sherlock(tmp_path)

    def describe(img, question):
        if said is None:
            raise RuntimeError("analyst endpoint down")
        return {"description": said}

    s.tools.analyst = SimpleNamespace(describe=describe)
    s.tools.verifier = SimpleNamespace(check=lambda img, claim: {
        "supported": yes, "reason": "Yes." if yes else "No.", "determinate": True})
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["verified"] is yes
    assert report["action"] == ("alert" if yes else "log")
    assert report["comparison"]["status"] == ("confirmed" if yes else "rejected")
    if yes and said == "Cake.":
        # #60: words that name another object never sit under a confirmed claim.
        assert report["description"] == "" and report["analyst_mismatch"] == "Cake."
    else:
        assert "analyst_mismatch" not in report and report["description"] == (said or "")
