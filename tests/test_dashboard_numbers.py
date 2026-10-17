"""Evidence selection tests use labelled fixtures, never publish model scores."""

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sherlock.dashboard import numbers
from sherlock.dashboard.server import make_app


def benchmark():
    def row(concurrency, p50, p95):
        return {
            "workload": "director",
            "model": "director",
            "concurrency": concurrency,
            "summary": {
                "succeeded": 200,
                "attempted": 200,
                "failed": 0,
                "e2e_ms": {"p50": p50, "p95": p95},
                "output_tokens_per_s": 42.5,
                "requests_per_s": 3.5,
            },
        }

    return {
        "evidence_kind": "endpoint",
        "prefix_caching": "on",
        "runs": [
            row(1, 100, 200),
            row(16, 300, 400),
            {"workload": "vlm", "concurrency": 32},
        ],
        "comparison": {
            "results": [
                {
                    "workload": "director",
                    "model": "director",
                    "concurrency": 1,
                    "baseline_failed": 0,
                    "candidate_failed": 0,
                    "e2e_p95_change_pct": -90,
                },
                {
                    "workload": "director",
                    "model": "director",
                    "concurrency": 16,
                    "baseline_failed": 0,
                    "candidate_failed": 0,
                    "e2e_p95_change_pct": -12.5,
                },
            ]
        },
    }


def ablation():
    return {
        "execution_mode": "live_endpoint",
        "ablation_id": "nemotron_lightning",
        "summary_metrics": {
            "num_tasks": 8,
            "task_success": 0.75,
            "tool_call_validity": 0.9,
        },
    }


def scout():
    return {
        "evidence_kind": "model_evaluation",
        "execution_mode": "measured",
        "split": "test",
        "images": {"room": 4},
        "before": {
            "model": "stock.pt",
            "room": {
                "weapon": {"tp": 2, "fp": 1, "fn": 2, "precision": 0.667, "recall": 0.5}
            },
        },
        "after": {
            "model": "tuned.pt",
            "room": {
                "weapon": {"tp": 3, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.75}
            },
        },
    }


def write(root, name, data, age=1):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    os.utime(path, (age, age))
    return path


def test_missing_and_checked_in_stub_results_are_tbd(tmp_path):
    assert numbers.read_numbers(tmp_path) == {
        "director": None,
        "ablation": None,
        "scout": None,
    }
    # The checked-in benchmark is explicitly protocol-only evidence.
    for path in (numbers.ROOT / "results").glob("vllm_bench_*.json"):
        data = json.loads(path.read_text())
        if data.get("evidence_kind") == "stub":
            assert numbers.director_result(data) is None


@pytest.mark.parametrize(
    "key, value",
    [
        ("evidence_kind", "stub"),
        ("evidence_kind", None),
        ("execution_mode", "local_stub"),
        ("stub", True),
        ("is_stub", True),
        ("mock", True),
    ],
)
def test_benchmark_stub_markers_override_other_labels(key, value):
    data = benchmark()
    data[key] = value
    assert numbers.director_result(data) is None


def test_latest_real_result_includes_nested_preflight_outputs(tmp_path):
    older = benchmark()
    older["runs"][1]["summary"]["e2e_ms"]["p50"] = 999
    write(tmp_path, "results/vllm_bench_old.json", older, 1)
    write(
        tmp_path, "results/brev_preflight_run/on/vllm_bench_live.json", benchmark(), 2
    )
    newer_stub = {**benchmark(), "evidence_kind": "stub"}
    write(tmp_path, "results/vllm_bench_new_stub.json", newer_stub, 3)
    result = numbers.read_numbers(tmp_path)["director"]
    assert result["file"] == "vllm_bench_live.json"
    assert (
        result["concurrency"] == 16
        and result["p50_ms"] == 300
        and result["p95_ms"] == 400
    )
    assert result["output_tokens_per_s"] == 42.5 and result["requests_per_s"] == 3.5
    assert result["prefix_cache_p95_change_pct"] == -12.5


@pytest.mark.parametrize(
    "change", ["missing", "model", "concurrency", "failed", "off", "stub"]
)
def test_ab_delta_requires_same_cell_and_valid_comparison(change):
    data = benchmark()
    row = data["comparison"]["results"][1]
    if change == "missing":
        data.pop("comparison")
    elif change == "model":
        row["model"] = "other-model"
    elif change == "concurrency":
        row["concurrency"] = 8
    elif change == "failed":
        row["baseline_failed"] = 1
    elif change == "off":
        data["prefix_caching"] = "off"
    else:
        row["evidence_kind"] = "stub"
    assert numbers.director_result(data)["prefix_cache_p95_change_pct"] is None


def test_latest_failed_run_does_not_fall_back_to_older_success(tmp_path):
    write(tmp_path, "results/vllm_bench_old.json", benchmark(), 1)
    data = benchmark()
    for row in data["runs"][:2]:
        row["summary"].update(
            succeeded=0,
            failed=200,
            e2e_ms={"p50": None, "p95": None},
            output_tokens_per_s=None,
            requests_per_s=0,
        )
    write(tmp_path, "results/vllm_bench_failed.json", data, 2)
    result = numbers.read_numbers(tmp_path)["director"]
    assert result["file"] == "vllm_bench_failed.json" and result["succeeded"] == 0
    assert result["p50_ms"] is None and result["prefix_cache_p95_change_pct"] is None


@pytest.mark.parametrize(
    "value", [None, True, "123", float("nan"), float("inf"), -1, 10**500]
)
def test_invalid_numbers_stay_missing(value):
    assert numbers.number(value) is None


def test_zero_values_are_preserved():
    assert numbers.number(0) == 0
    data = ablation()
    data["summary_metrics"]["task_success"] = 0
    assert numbers.ablation_result(data)["task_success"] == 0


def test_ablation_requires_live_mode_and_labels_reference(tmp_path):
    write(tmp_path, "results/ablation_live.json", ablation(), 1)
    write(
        tmp_path,
        "results/ablation_stub.json",
        {**ablation(), "execution_mode": "local_stub"},
        2,
    )
    result = numbers.read_numbers(tmp_path)["ablation"]
    assert result["file"] == "ablation_live.json" and result["tasks"] == 8
    assert result["task_success"] == 0.75
    assert numbers.ablation_result({**ablation(), "execution_mode": None}) is None
    other = {**ablation(), "ablation_id": "other"}
    other["summary_metrics"] = {**other["summary_metrics"], "task_success": 1}
    summary = {"execution_mode": "live_endpoint", "results": [other, ablation()]}
    assert numbers.ablation_result(summary)["config"] == "nemotron_lightning"
    summary["execution_mode"] = "local_stub"
    assert numbers.ablation_result(summary) is None


def test_scout_provenance_split_and_matched_ground_truth():
    data = scout()
    result = numbers.scout_result(data)
    assert result["source"] == "room" and result["boxes"] == 4
    assert result["before_recall"] == 0.5 and result["after_recall"] == 0.75
    for overrides in (
        {"evidence_kind": "stub"},
        {"evidence_kind": None},
        {"execution_mode": "smoke"},
        {"execution_mode": "local_stub"},
        {"execution_mode": "unknown"},
        {"execution_mode": None},
        {"split": "train"},
        {"mock": True},
    ):
        assert numbers.scout_result({**data, **overrides}) is None
    data["after"]["room"]["weapon"]["fn"] = 0
    assert numbers.scout_result(data) is None


def test_scout_all_fallback_and_supported_locations(tmp_path):
    data = scout()
    for side in ("before", "after"):
        data[side]["all"] = data[side].pop("room")
    write(tmp_path, "runs/scout/eval.json", data)
    assert numbers.read_numbers(tmp_path)["scout"]["source"] == "all"
    write(tmp_path, "results/scout_eval_latest.json", scout(), 2)
    assert numbers.read_numbers(tmp_path)["scout"]["source"] == "room"


@pytest.mark.parametrize("missing", [("evidence_kind",), ("execution_mode",), ("evidence_kind", "execution_mode")])
def test_scout_unlabelled_reports_are_not_evidence(missing):
    data = scout()
    data.update(conf=0.35, iou=0.5)
    for key in missing:
        data.pop(key)
    assert numbers.scout_result(data) is None


def test_scout_newer_smoke_is_skipped_and_nested_smoke_is_rejected(tmp_path):
    write(tmp_path, "results/scout_eval_measured.json", scout())
    data = {**scout(), "execution_mode": "smoke"}
    write(tmp_path, "runs/scout/eval.json", data, 2)
    assert numbers.read_numbers(tmp_path)["scout"]["file"] == "scout_eval_measured.json"
    for side in ("before", "after"):
        data = scout()
        data[side]["execution_mode"] = "smoke"
        assert numbers.scout_result(data) is None


def test_malformed_and_oversized_files_do_not_break_dashboard(tmp_path, monkeypatch):
    path = write(tmp_path, "results/vllm_bench_bad.json", {})
    for raw in (
        "{incomplete",
        "null",
        "[]",
        '{"evidence_kind": []}',
        '{"evidence_kind":"endpoint","runs":{}}',
    ):
        path.write_text(raw)
        assert numbers.read_numbers(tmp_path)["director"] is None
    path.write_bytes(b"\xff\xfe\x00")
    assert numbers.read_numbers(tmp_path)["director"] is None
    write(tmp_path, "results/vllm_bench_large.json", benchmark())
    monkeypatch.setattr(numbers, "MAX_BYTES", 16)
    assert numbers.read_numbers(tmp_path)["director"] is None


def test_symlink_artifacts_are_not_read(tmp_path):
    target = write(tmp_path, "outside.json", benchmark())
    results = tmp_path / "results"
    results.mkdir()
    (results / "vllm_bench_link.json").symlink_to(target)
    assert numbers.read_numbers(tmp_path)["director"] is None


def test_api_reads_updated_files_and_returns_only_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(numbers, "ROOT", tmp_path)
    with TestClient(make_app(SimpleNamespace())) as client:
        assert client.get("/numbers").json() == {
            "director": None,
            "ablation": None,
            "scout": None,
        }
        data = benchmark()
        data["director_base_url"] = "https://secret-host/private"
        write(tmp_path, "results/vllm_bench_live.json", data)
        write(tmp_path, "results/ablation_live.json", ablation())
        write(tmp_path, "runs/scout/eval.json", scout())
        response = client.get("/numbers")
        assert (
            response.status_code == 200
            and response.headers["cache-control"] == "no-store"
        )
        assert "secret-host" not in response.text
        assert response.json()["director"]["concurrency"] == 16
        assert response.json()["ablation"]["tasks"] == 8
        assert response.json()["scout"]["boxes"] == 4
        assert client.get("/numbers.js").status_code == 200
        page = client.get("/log").text
        assert 'id="avoided"' in page


def test_no_producer_objects_are_mutated():
    data = benchmark()
    original = copy.deepcopy(data)
    numbers.director_result(data)
    assert data == original
