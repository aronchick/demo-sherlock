"""Bounded summaries of measured results. Missing or untrusted evidence stays null.

Read the newest eligible artifact by modification time (filename breaks ties).
The Director headline uses the highest measured concurrency, never the fastest
row. A/B deltas must match that same model/concurrency and have no failed calls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAX_BYTES = 16 * 1024 * 1024


def number(value, *, minimum=0, maximum=None):
    if type(value) not in (int, float):
        return None
    try:
        if not math.isfinite(value):
            return None
    except OverflowError:
        return None
    if value < minimum or maximum is not None and value > maximum:
        return None
    return value


def count(value):
    return value if type(value) is int and value >= 0 else None


def text(value):
    return value[:120] if isinstance(value, str) else ""


def mapping(value):
    return value if isinstance(value, dict) else {}


def no_stub(data):
    """A conflicting stub/mock marker always wins over a positive label."""
    return not (
        data.get("stub") is True
        or data.get("is_stub") is True
        or data.get("mock") is True
        or data.get("evidence_kind") in {"stub", "mock", "synthetic"}
        or data.get("execution_mode") in {"local_stub", "stub", "mock", "synthetic", "smoke"}
    )


def director_result(data):
    if data.get("evidence_kind") != "endpoint" or not no_stub(data):
        return None
    runs = data.get("runs")
    if not isinstance(runs, list):
        return None
    candidates = []
    for row in runs:
        row = mapping(row)
        summary = mapping(row.get("summary"))
        concurrency, succeeded = (
            count(row.get("concurrency")),
            count(summary.get("succeeded")),
        )
        attempted = count(summary.get("attempted"))
        if (
            row.get("workload") == "director"
            and no_stub(row)
            and concurrency
            and succeeded is not None
            and attempted
            and succeeded <= attempted
        ):
            candidates.append(row)
    if not candidates:
        return None
    row = max(candidates, key=lambda item: item["concurrency"])
    summary = row["summary"]
    latency = mapping(summary.get("e2e_ms"))
    delta = None
    comparison = mapping(data.get("comparison"))
    comparisons = comparison.get("results")
    if (
        data.get("prefix_caching") == "on"
        and isinstance(data.get("workload_fingerprint"), str)
        and bool(data["workload_fingerprint"])
        and comparison.get("baseline_workload_fingerprint") == data["workload_fingerprint"]
        and comparison.get("candidate_workload_fingerprint") == data["workload_fingerprint"]
        and isinstance(comparisons, list)
        and no_stub(comparison)
        and count(summary.get("failed")) == 0
    ):
        for item in comparisons:
            item = mapping(item)
            if (
                item.get("workload") == "director"
                and item.get("model") == row.get("model")
                and item.get("concurrency") == row["concurrency"]
                and no_stub(item)
                and count(item.get("baseline_failed")) == 0
                and count(item.get("candidate_failed")) == 0
            ):
                delta = number(item.get("e2e_p95_change_pct"), minimum=-100)
                break
    return {
        "model": text(row.get("model")),
        "concurrency": row["concurrency"],
        "succeeded": summary["succeeded"],
        "attempted": summary["attempted"],
        "p50_ms": number(latency.get("p50")),
        "p95_ms": number(latency.get("p95")),
        "output_tokens_per_s": number(summary.get("output_tokens_per_s")),
        "requests_per_s": number(summary.get("requests_per_s")),
        "prefix_cache_p95_change_pct": delta,
    }


def ablation_result(data):
    if data.get("execution_mode") != "live_endpoint" or not no_stub(data):
        return None
    rows = data.get("results", [data])
    if not isinstance(rows, list):
        return None
    # A summary file contains several configurations. Prefer the named reference,
    # then configuration ID, so the panel does not cherry-pick the best score.
    rows = sorted(
        (r for r in rows if isinstance(r, dict)),
        key=lambda r: (
            r.get("ablation_id") != "nemotron_lightning",
            text(r.get("ablation_id")),
        ),
    )
    for row in rows:
        metrics = mapping(row.get("summary_metrics"))
        tasks = count(metrics.get("num_tasks"))
        if not no_stub(row) or not tasks:
            continue
        return {
            "config": text(row.get("ablation_id")),
            "tasks": tasks,
            "task_success": number(metrics.get("task_success"), maximum=1),
            "tool_call_validity": number(metrics.get("tool_call_validity"), maximum=1),
        }
    return None


def scout_result(data):
    # A complete legacy report can still come from the tiny CPU smoke test.
    # Require both producer labels; model scores alone are not provenance.
    if (
        data.get("evidence_kind") != "model_evaluation"
        or data.get("execution_mode") != "measured"
        or not no_stub(data)
    ):
        return None
    if data.get("split") not in {"test", "val"}:
        return None
    before, after = mapping(data.get("before")), mapping(data.get("after"))
    baseline_only = data.get("evaluation_scope") == "baseline_only" and data.get("after") is None
    if (
        not text(before.get("model"))
        or (not baseline_only and not text(after.get("model")))
        or not no_stub(before)
        or not no_stub(after)
    ):
        return None
    for source in ("room", "coco", "all"):
        b = mapping(mapping(before.get(source)).get("weapon"))
        a = mapping(mapping(after.get(source)).get("weapon"))
        counts = [count(row.get(key)) for row in ((b,) if baseline_only else (b, a)) for key in ("tp", "fn")]
        if None in counts or not sum(counts[:2]) or (not baseline_only and sum(counts[:2]) != sum(counts[2:])):
            continue
        return {
            "source": source,
            "environment": text(data.get("environment")),
            "device": text(data.get("device")),
            "split": data["split"],
            "boxes": sum(counts[:2]),
            "before_precision": number(b.get("precision"), maximum=1),
            "after_precision": number(a.get("precision"), maximum=1),
            "before_recall": number(b.get("recall"), maximum=1),
            "after_recall": number(a.get("recall"), maximum=1),
        }
    return None


def latest(root, paths, parse):
    candidates = []
    for path in paths:
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                continue
            stat = path.stat()
            if stat.st_size <= MAX_BYTES:
                candidates.append((stat.st_mtime_ns, path.as_posix(), path))
        except OSError:
            continue
    for _, _, path in sorted(candidates, reverse=True):
        try:
            with path.open() as stream:
                raw = stream.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                continue
            data = json.loads(raw)
            result = parse(data) if isinstance(data, dict) else None
        except (OSError, UnicodeError, ValueError, TypeError, KeyError):
            continue
        if result is not None:
            return {**result, "file": path.name}
    return None


def read_numbers(root=None):
    root = ROOT if root is None else Path(root)
    results = root / "results"
    return {
        "director": latest(root, results.rglob("vllm_bench_*.json"), director_result),
        "ablation": latest(root, results.glob("ablation_*.json"), ablation_result),
        "scout": latest(
            root,
        [root / "runs/scout/eval.json", root / "runs/scout/eval_baseline.json", *results.glob("scout_eval*.json")],
            scout_result,
        ),
    }
