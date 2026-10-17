"""Explicit Analyst verdicts gate alerts; descriptive keywords never do."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sherlock.agents.vision import Analyst, parse_assessment
from sherlock.config import Endpoint
from test_weapon_flow import _frozen_sherlock, _knife_det


@pytest.mark.parametrize("text", [
    "A spoon rather than a knife.", "A toy knife.", "A knife.", "", "yes", "[]",
    '{"supported":"yes","description":"A knife."}',
    '{"claim":"a cup","supported":"yes","description":"A knife."}',
    '{"claim":"a knife","supported":true,"description":"A knife."}',
    '{"claim":"a knife","supported":"probably","description":"A knife."}',
])
def test_unstructured_or_invalid_analyst_response_never_supports(text):
    assert parse_assessment(text, "a knife")["supported"] is None


@pytest.mark.parametrize(("verdict", "expected"), [("yes", True), ("no", False), ("uncertain", None)])
def test_structured_assessment_is_bound_to_claim(verdict, expected):
    raw = json.dumps({"claim": "a knife", "supported": verdict, "description": "Object description."})
    result = parse_assessment(raw, "a knife")
    assert result["supported"] is expected
    assert result["description"] == "Object description."


@pytest.mark.parametrize("description", ["A spoon rather than a knife.", "A toy knife."])
def test_rejected_contrast_or_toy_cannot_alert_even_with_positive_verifier(description, tmp_path):
    s = _frozen_sherlock(tmp_path)
    result = parse_assessment(json.dumps({"claim": "a knife", "supported": "no",
                                         "description": description}), "a knife")
    s.tools.analyst = SimpleNamespace(assess=lambda *args: result)
    s.tools.verifier = SimpleNamespace(check=lambda *args: {"supported": True, "reason": "yes"})
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["action"] == "log" and not report["verified"]
    assert report["comparison"]["status"] == "disagreed"
    assert report["description"] == description


def test_analyst_assessment_uses_one_request_and_preserves_description(monkeypatch):
    requests = []
    result = json.dumps({"claim": "a knife", "supported": "yes", "description": "A steel knife."})

    def request(client, model, img, prompt, max_tokens=200):
        requests.append(prompt)
        return result

    monkeypatch.setattr("sherlock.agents.vision._vision_call", request)
    analyst = Analyst(Endpoint("http://unused/v1", "model"))
    out = analyst.assess(np.zeros((32, 32, 3), dtype=np.uint8), "a knife")
    assert len(requests) == 1
    assert '"a knife"' in requests[0] and "replica" in requests[0]
    assert out["supported"] is True and out["description"] == "A steel knife."


@pytest.mark.parametrize("raw", ["A toy knife.", "A spoon rather than a knife.", "Yes", "{}"])
def test_unparseable_assessment_routes_to_review(raw, tmp_path):
    s = _frozen_sherlock(tmp_path)
    s.tools.analyst = SimpleNamespace(assess=lambda *args: parse_assessment(raw, "a knife"))
    s.tools.verifier = SimpleNamespace(check=lambda *args: {"supported": True, "reason": "yes"})
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["comparison"]["analyst_supports"] is None
    assert report["comparison"]["status"] == "uncertain"
    assert report["action"] == "log" and not report["verified"]
