"""Verifier against a loopback OpenAI-compatible stub: the verdict is decided from logprobs
under a JSON schema, and endpoints that refuse or ignore those features fall back one mode
at a time (vLLM, then llama.cpp/Ollama-style, then plain text)."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import openai
import pytest
from sherlock.agents.vision import MARGIN_DEFAULTS, VERDICT_SCHEMA, Verifier
from sherlock.config import Endpoint
from sherlock.events import BUS

IMG = np.zeros((48, 64, 3), np.uint8)
SMALL, LARGE = "HuggingFaceTB/SmolVLM-500M-Instruct", "HuggingFaceTB/SmolVLM-Instruct"


def entry(token, logprob, top=()):
    return {"token": token, "logprob": logprob, "bytes": None,
            "top_logprobs": [{"token": t, "logprob": lp, "bytes": None} for t, lp in top]}


def schema_tokens(verdict="true", top=((" true", -0.05), (" false", -3.05))):
    """vLLM-shaped logprobs for {"supported": <verdict>, ...}, split the way a BPE would."""
    lp = dict(top).get(f" {verdict}", -0.1)
    return [entry('{"', 0.0), entry("supported", 0.0), entry('":', 0.0), entry(f" {verdict}", lp, top),
            entry(', "confidence": 0.99, "reason": "stub"}', 0.0)]


class Stub:
    """What the endpoint does with the Verifier's request fields."""

    def __init__(self):
        self.content = '{"supported": true, "confidence": 0.99, "reason": "stub"}'
        self.tokens = schema_tokens()
        self.refuse: dict[str, tuple[int, str]] = {}  # request field -> (status, message)
        self.ignore: set[str] = set()                  # request fields accepted but not honoured
        self.status = 200
        self.bodies: list[dict] = []


@pytest.fixture
def stub():
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.bodies.append(body)
            for field, (status, msg) in state.refuse.items():
                if field in body:
                    return self.reply(status, {"error": {"message": msg, "type": "invalid_request"}})
            if state.status != 200:
                return self.reply(state.status, {"error": {"message": "model not found"}})
            lp = ({"content": state.tokens, "refusal": None}
                  if body.get("logprobs") and "logprobs" not in state.ignore else None)
            self.reply(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                             "choices": [{"index": 0, "finish_reason": "stop", "logprobs": lp,
                                          "message": {"role": "assistant", "content": state.content}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()


def verifier(stub, model=SMALL):
    return Verifier(Endpoint(stub.url, model, "unused"))


@pytest.mark.parametrize(("model", "expected"), [(SMALL, False), (LARGE, True)])
def test_margin_mode_decides_from_logprobs_against_the_fitted_threshold(stub, model, expected):
    out = verifier(stub, model).check(IMG, "a knife")
    assert out["mode"] == "margin" and out["margin"] == pytest.approx(3.0)
    # 3.0 clears the 2.2B's 0.56 but not the 500M's 3.31, although both generated "true".
    assert out["supported"] is expected
    assert out["confidence"] == pytest.approx(1 / (1 + math.exp(-3.0)), abs=1e-3)
    body = stub.bodies[-1]
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}}
    assert next(iter(body["response_format"]["json_schema"]["schema"]["properties"])) == "supported"
    assert body["logprobs"] is True and body["top_logprobs"] >= 2
    assert body["messages"][0]["content"][1]["text"] == Verifier.PROMPT.format(claim="a knife")


def test_verifier_margin_env_overrides_the_fitted_default(stub, monkeypatch):
    monkeypatch.setenv("VERIFIER_MARGIN", "2.5")
    assert MARGIN_DEFAULTS[SMALL] > 3.0
    assert verifier(stub, SMALL).check(IMG, "a knife")["supported"] is True


def test_unlisted_model_decides_by_argmax(stub):
    stub.tokens = schema_tokens("true", ((" true", -0.9), (" false", -0.4)))
    out = verifier(stub, "smolvlm:500m").check(IMG, "a knife")
    assert out["mode"] == "margin" and out["margin"] == pytest.approx(-0.5)
    assert out["supported"] is False, "the generated 'true' is overruled by the logprobs"


def test_margin_pools_spellings_across_a_separate_whitespace_token(stub):
    stub.tokens = [entry('{"supported":', 0.0), entry(" ", 0.0),
                   entry("true", -1.0, (("true", -1.0), (" true", -1.0), ("false", -0.9))),
                   entry(', "confidence": 1, "reason": "x"}', 0.0)]
    out = verifier(stub, LARGE).check(IMG, "a knife")
    assert out["margin"] == pytest.approx(-1.0 + math.log(2) + 0.9, abs=1e-3)


def test_side_missing_from_top_logprobs_is_bounded_by_the_floor(stub):
    stub.tokens = schema_tokens("true", ((" true", -0.01), (" ,", -6.0)))
    assert verifier(stub).check(IMG, "a knife")["margin"] == pytest.approx(5.99)


@pytest.mark.parametrize(("content", "expected"), [
    ('{"supported": "false", "confidence": 0.9, "reason": "a cup"}', False),  # a string is not truthy
    ('{"supported": true, "confidence": 0.9, "reason": "a long reason cut off by max_to', True),
])
def test_constrained_mode_uses_the_generated_boolean_without_logprobs(stub, content, expected):
    stub.ignore.add("logprobs")  # e.g. an Ollama build without logprobs
    stub.content = content
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "constrained" and out["margin"] is None
    assert out["supported"] is expected


def test_refused_logprobs_are_dropped_once_and_remembered(stub):
    stub.refuse["logprobs"] = (400, "logprobs is not supported by this server")
    v = verifier(stub)
    assert v.check(IMG, "a knife")["mode"] == "constrained"
    assert [("logprobs" in b, "response_format" in b) for b in stub.bodies] == [(True, True), (False, True)]
    v.check(IMG, "a knife")
    assert len(stub.bodies) == 3 and "logprobs" not in stub.bodies[-1]


@pytest.mark.parametrize(("content", "expected"), [
    ("Supports.", True), ("True.", True), ("Answer: true", True), ("Yes, a knife.", True),
    ("No, it is a cup.", False), ("false.", False), ("The image does not show a kitchen knife.", False),
    ("0-1", False), ('{"supported": true, "confidence": 0.9, "reason": "knife"}', True),
])
def test_refused_schema_falls_back_to_a_lenient_text_read(stub, content, expected):
    stub.refuse["response_format"] = (422, "response_format: json_schema is not supported")
    stub.content = content
    stub.tokens = [entry(content, -0.1, ((content, -0.1), ("x", -5.0)))]
    v = verifier(stub)
    out = v.check(IMG, "a knife")
    assert out["mode"] == "text" and out["supported"] is expected
    assert not v.structured and v.logprobs, "only the refused feature is dropped"


def test_ignored_schema_reads_text(stub):
    stub.ignore.add("logprobs")  # an MLX-style server that accepts and ignores both fields
    stub.content = "True."
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "text" and out["supported"] is True


def test_other_errors_propagate_and_keep_the_capabilities(stub):
    stub.status = 404
    v = verifier(stub)
    with pytest.raises(openai.NotFoundError):
        v.check(IMG, "a knife")
    assert v.structured and v.logprobs


def test_verdict_event_records_the_mode(stub):
    q = BUS.subscribe()
    verifier(stub, LARGE).check(IMG, "a knife")
    ev = next(e for e in iter(q.get_nowait, None) if e.stage == "verifier")
    assert ev.data["mode"] == "margin" and ev.data["margin"] == pytest.approx(3.0)
    assert "margin +3.00 vs 0.56" in ev.msg
