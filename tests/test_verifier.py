"""Verifier against a loopback OpenAI-compatible stub: a yes/no question, decided from the
logprobs of the reply's first word when the endpoint returns them (vLLM, llama.cpp) and from
the first word itself when it refuses or ignores them (Ollama, MLX)."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import openai
import pytest
from sherlock.agents.vision import Verifier
from sherlock.config import Endpoint
from sherlock.events import BUS

IMG = np.zeros((48, 64, 3), np.uint8)
MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


def entry(token, logprob, top=()):
    return {"token": token, "logprob": logprob, "bytes": None,
            "top_logprobs": [{"token": t, "logprob": lp, "bytes": None} for t, lp in top]}


def answer(word=" Yes", top=((" Yes", -0.05), (" No", -3.05))):
    """vLLM-shaped logprobs for a reply such as ' Yes.'"""
    return [entry(word, dict(top).get(word, -0.1), top), entry(".", 0.0)]


class Stub:
    """What the endpoint does with the Verifier's request fields."""

    def __init__(self):
        self.content = "Yes."
        self.tokens = answer()
        self.refuse: dict[str, tuple[int, str]] = {}  # request field -> (status, message)
        self.ignore: set[str] = set()                  # request fields accepted but not honoured
        self.status = 200
        self.logprobs = None  # sent verbatim instead of {"content": tokens} when set
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
            if state.logprobs is not None:
                lp = state.logprobs
            self.reply(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                             "choices": [{"index": 0, "finish_reason": "stop", "logprobs": lp,
                                          "message": {"role": "assistant", "content": state.content}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def default_threshold(monkeypatch):
    monkeypatch.delenv("VERIFIER_MARGIN", raising=False)


def verifier(stub):
    return Verifier(Endpoint(stub.url, MODEL, "unused"))


@pytest.mark.parametrize(("content", "tokens", "margin", "expected"), [
    ("Yes.", answer(), 3.0, True),
    ("No.", answer(" No", ((" No", -0.1), (" Yes", -2.1))), -2.0, False),
])
def test_margin_mode_decides_by_argmax_by_default(stub, content, tokens, margin, expected):
    stub.content, stub.tokens = content, tokens
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "margin" and out["margin"] == pytest.approx(margin)
    assert out["supported"] is expected and out["reason"] == content
    assert out["confidence"] == pytest.approx(1 / (1 + math.exp(-margin)), abs=1e-3)
    body = stub.bodies[-1]
    assert "response_format" not in body
    assert body["logprobs"] is True and body["top_logprobs"] >= 2
    assert body["messages"][0]["content"][1]["text"] == (
        "Is the object in this image a knife? Judge the object only; do not identify or describe "
        "any person. Answer yes or no.")


def test_logprobs_overrule_the_sampled_word(stub):
    stub.tokens = answer(" Yes", ((" Yes", -0.9), (" No", -0.4)))
    out = verifier(stub).check(IMG, "a knife")
    assert out["margin"] == pytest.approx(-0.5) and out["supported"] is False


def test_verifier_margin_env_sets_the_threshold(stub, monkeypatch):
    stub.content, stub.tokens = "No.", answer(" No", ((" No", -0.1), (" Yes", -2.1)))
    assert verifier(stub).check(IMG, "a knife")["supported"] is False
    monkeypatch.setenv("VERIFIER_MARGIN", "-2.75")  # the 500M's bake-off fit
    out = verifier(stub).check(IMG, "a knife")
    assert out["supported"] is True and out["margin"] == pytest.approx(-2.0)


def test_spellings_pool_and_leading_whitespace_is_skipped(stub):
    stub.tokens = [entry(" ", 0.0), entry("Yes", -1.0, (("Yes", -1.0), (" yes", -1.0), ("No", -0.9))),
                   entry(".", 0.0)]
    out = verifier(stub).check(IMG, "a knife")
    assert out["margin"] == pytest.approx(-1.0 + math.log(2) + 0.9, abs=1e-3)


def test_sampled_word_missing_from_top_logprobs_still_counts(stub):
    stub.content = "No."
    stub.tokens = [entry(" No", -0.2, ((" Yes", -3.0), (" Maybe", -4.0))), entry(".", 0.0)]
    assert verifier(stub).check(IMG, "a knife")["margin"] == pytest.approx(-2.8)


def test_side_missing_from_top_logprobs_is_bounded_by_the_floor(stub):
    stub.tokens = answer(" Yes", ((" Yes", -0.01), (" It", -6.0)))
    assert verifier(stub).check(IMG, "a knife")["margin"] == pytest.approx(5.99)


def test_extreme_margin_is_clamped(stub):
    stub.tokens = answer(" Yes", ((" Yes", 0.0), (" No", -1e6)))
    out = verifier(stub).check(IMG, "a knife")
    assert out["margin"] == 100.0 and out["confidence"] == 1.0


def test_reply_not_opening_with_yes_or_no_is_not_support(stub):
    stub.content = "The object is a knife."
    stub.tokens = [entry("The", -0.1, (("The", -0.1), ("Yes", -3.0))), entry(" object is a knife.", 0.0)]
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "text" and out["margin"] is None and out["supported"] is False


@pytest.mark.parametrize(("content", "expected"), [
    ("Yes.", True), ("yes, it is a knife", True), ("Answer: yes", True), ("True.", True),
    ("No, it is a cup.", False), ("no", False), ("The object is a knife.", False), ("", False),
])
def test_text_mode_reads_the_first_word_when_logprobs_are_ignored(stub, content, expected):
    stub.ignore.add("logprobs")  # e.g. an Ollama or mlx-vlm server
    stub.content = content
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "text" and out["margin"] is None and out["confidence"] is None
    assert out["supported"] is expected


def test_refused_logprobs_are_dropped_once_and_remembered(stub):
    stub.refuse["logprobs"] = (400, "logprobs is not supported by this server")
    v = verifier(stub)
    out = v.check(IMG, "a knife")
    assert out["mode"] == "text" and out["supported"] is True
    assert ["logprobs" in b for b in stub.bodies] == [True, False]
    v.check(IMG, "a knife")
    assert len(stub.bodies) == 3 and "logprobs" not in stub.bodies[-1]


def test_nonstandard_logprobs_fall_back_to_the_first_word(stub):
    stub.tokens = [entry(" Yes", None, ((" Yes", None), (" No", None)))]
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "text" and out["supported"] is True


@pytest.mark.parametrize(("content", "expected"), [("Yes.", True), ("No.", False)])
def test_mlx_vlm_placeholder_logprobs_read_the_first_word(stub, content, expected):
    # mlx-vlm 0.7.8 accepts logprobs without an error, then sends 0.0 for every token and no
    # alternatives; read as logprobs, 0.0 would look like certainty.
    stub.content = content
    stub.tokens = [entry(f" {content[:-1]}", 0.0), entry(".", 0.0)]
    v = verifier(stub)
    out = v.check(IMG, "a knife")
    assert out["mode"] == "text" and out["margin"] is None and out["supported"] is expected
    assert v.logprobs and stub.bodies[-1]["logprobs"] is True


@pytest.mark.parametrize("payload", [[0.0, 0.0], "n/a"])
def test_logprobs_that_are_not_an_object_fall_back_to_the_first_word(stub, payload):
    stub.logprobs = payload
    out = verifier(stub).check(IMG, "a knife")
    assert out["mode"] == "text" and out["supported"] is True


def test_other_errors_propagate_and_keep_logprobs(stub):
    stub.status = 404
    v = verifier(stub)
    with pytest.raises(openai.NotFoundError):
        v.check(IMG, "a knife")
    assert v.logprobs


def test_verdict_event_records_the_mode(stub):
    q = BUS.subscribe()
    verifier(stub).check(IMG, "a knife")
    ev = next(e for e in iter(q.get_nowait, None) if e.stage == "verifier")
    assert ev.data["mode"] == "margin" and ev.data["margin"] == pytest.approx(3.0)
    assert ev.msg == "CONFIRMED (margin +3.00 vs +0.00): Yes."
