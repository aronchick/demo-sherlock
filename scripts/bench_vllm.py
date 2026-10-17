# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27,<1", "jsonschema>=4.23,<5", "numpy>=1.26",
#   "opencv-python>=4.10", "openai>=1.40", "python-dotenv>=1",
#   "prometheus-client>=0.21,<1",
# ]
# ///
"""Measure Sherlock-shaped streaming traffic, without executing camera tools.

Run with uv run scripts/bench_vllm.py. Endpoint credentials stay in environment
variables; neither response text nor endpoint URLs are written to the results.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import httpx
import numpy as np
from jsonschema import Draft202012Validator, ValidationError
from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sherlock"))

from sherlock.agents import director
from sherlock.agents.vision import Analyst, Verifier
from sherlock.tools import TOOL_SCHEMAS

SYSTEM = director.SYSTEM
SCHEMA_VERSION = 1
FIXTURE = ROOT / "tests/fixtures/vllm/shuttle_crop.jpg"
PARAMETERS = {
    tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOL_SCHEMAS
}
VALIDATORS = {name: Draft202012Validator(schema) for name, schema in PARAMETERS.items()}


def digest(value) -> str:
    raw = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True).encode()
    )
    return hashlib.sha256(raw).hexdigest()


def load_crop(path: Path) -> tuple[str, dict]:
    raw = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if not raw.startswith(b"\xff\xd8") or img is None:
        raise ValueError("--image must be a decodable JPEG photograph crop")
    height, width = img.shape[:2]
    # Match the application's 768px, quality-85 image preprocessing.
    scale = min(1, 768 / max(height, width))
    if scale < 1:
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode JPEG crop")
    wire = encoded.tobytes()
    return base64.b64encode(wire).decode(), {
        "source_sha256": digest(raw),
        "wire_sha256": digest(wire),
        "width": img.shape[1],
        "height": img.shape[0],
        "bytes": len(wire),
    }


def tool_arguments(name: str, values: dict) -> dict:
    """Keep fixtures compatible with the imported schema; reject missing fields."""
    args = {k: v for k, v in values.items() if k in PARAMETERS[name]["properties"]}
    try:
        VALIDATORS[name].validate(args)
    except ValidationError as exc:
        raise ValueError(
            f"Benchmark fixture no longer matches {name}: {exc.message}"
        ) from exc
    return args


def director_case(index: int) -> tuple[dict, list]:
    """Synthetic tool results for each branch of the real investigation flow."""
    threat = ("weapon", "unknown", "benign")[index % 3]
    x = 120 + index % 320
    detection = {
        "camera": "hero",
        "label": {"weapon": "knife", "unknown": "object", "benign": "cup"}[threat],
        "threat": threat,
        "head_bbox": [x, 20, x + 80, 100]
        if threat == "weapon" and index // 18 % 2 == 0
        else None,
        "confidence": round(0.8 + index % 19 / 100, 2),
        "bbox": [x, 180, x + 160, 320],
        "cameras": [
            {"name": name, "pan": 0, "tilt": 0, "zoom": 1, "covered": False}
            for name in ("hero", "scout1")
        ],
    }
    history = []

    def step(name, values, result):
        history.append((name, tool_arguments(name, values), result))

    evidence = []
    claim = "benign cup"
    if threat != "benign":
        pose = {
            "ok": True,
            "name": "hero",
            "pan": -30,
            "tilt": 5,
            "zoom": 3,
            "covered": False,
        }
        step("look_at", {"camera": "hero", "bbox": detection["bbox"]}, pose)
        subject = "weapon" if threat == "weapon" else "object"
        claim = (
            "A kitchen knife with a steel blade and a dark handle."
            if threat == "weapon"
            else "A white model spacecraft with dark windows."
        )
        evidence.append(f"{subject}_crop")
        step(
            "snapshot",
            {"camera": "hero", "subject": subject, "question": "What is this object?"},
            {"description": claim, "evidence_id": evidence[0]},
        )
        # Permit execution before the weapon-flow PR lands, using the installed
        # schema's capabilities. Face evidence is metadata only, never VLM input.
        if detection["head_bbox"] and "subject" in PARAMETERS["snapshot"]["properties"]:
            step("look_at", {"camera": "hero", "bbox": detection["head_bbox"]}, pose)
            step(
                "snapshot",
                {"camera": "hero", "subject": "face"},
                {"evidence_id": "face_crop", "stored": True, "analyzed": False},
            )
            evidence.append("face_crop")
        if "verify" in PARAMETERS:
            step(
                "verify",
                {"claim": claim},
                {
                    "supported": True,
                    "confidence": 0.9,
                    "reason": "Object visible in crop.",
                },
            )
    step(
        "report",
        {
            "claim": claim,
            "confidence": 0.9,
            "evidence": evidence,
            "action": {"weapon": "alert", "unknown": "log", "benign": "ignore"}[threat],
        },
        {"done": True},
    )
    return detection, history


def payload(workload: str, index: int, model: str, crop: str, max_calls: int) -> dict:
    if workload == "director":
        detection, history = director_case(index)
        budget = max_calls + (
            getattr(director, "WEAPON_EXTRA_CALLS", 4)
            if detection["threat"] == "weapon"
            else 0
        )
        messages = [
            {"role": "system", "content": SYSTEM.format(max_calls=budget)},
            {"role": "user", "content": json.dumps(detection)},
        ]
        # Sample requests before each step, including the final report. Never
        # request another action after a completed report. Benign stays one turn.
        for turn, (name, args, result) in enumerate(
            history[: index // 3 % len(history)]
        ):
            call_id = f"call_{turn}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result),
                    },
                ]
            )
        return {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    prompt = (
        Analyst.PROMPT.format(question="What is this object? Read any visible text.")
        if index % 2 == 0
        else Verifier.PROMPT.format(claim="a white model spacecraft with dark windows")
    )
    return {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 200 if index % 2 == 0 else 120,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{crop}"},
                    },
                ],
            }
        ],
    }


def reject_nonfinite(value):
    raise ValueError(f"Non-JSON number: {value}")


def valid_call(call: dict) -> bool:
    try:
        name = call["name"]
        if (
            not call.get("id")
            or call.get("type") != "function"
            or name not in PARAMETERS
        ):
            return False
        args = json.loads(call["arguments"], parse_constant=reject_nonfinite)
        VALIDATORS[name].validate(args)
        # The dispatcher accepts named Python arguments, not arbitrary extra keys.
        return isinstance(args, dict) and not (
            args.keys() - PARAMETERS[name]["properties"].keys()
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        return False


async def sse_events(response: httpx.Response):
    data = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


async def request(
    client: httpx.AsyncClient, url: str, body: dict, timeout: float
) -> dict:
    sample = {
        "ok": False,
        "error": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "finish_reason": None,
        "tool_calls": 0,
        "valid_tool_calls": 0,
        "tool_response_valid": False,
    }
    calls = {}
    start = time.perf_counter()
    done = False
    try:
        # Wall-clock deadline includes the full stream, unlike an idle-read timeout.
        async with asyncio.timeout(timeout):
            async with client.stream(
                "POST",
                url,
                json={
                    **body,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                response.raise_for_status()
                async for event in sse_events(response):
                    if event == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(event)
                    if "error" in chunk:
                        raise ValueError("stream error")
                    usage = chunk.get("usage") or {}
                    for source, target in (
                        ("completion_tokens", "output_tokens"),
                        ("prompt_tokens", "prompt_tokens"),
                    ):
                        count = usage.get(source)
                        if type(count) is int and count >= 0:
                            sample[target] = count
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        fragments = delta.get("tool_calls") or []
                        meaningful = (
                            delta.get("content")
                            or delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or any(
                                f.get("function", {}).get("name")
                                or f.get("function", {}).get("arguments")
                                for f in fragments
                            )
                        )
                        if meaningful and sample["ttft_ms"] is None:
                            sample["ttft_ms"] = (time.perf_counter() - start) * 1000
                        for fragment in fragments:
                            call = calls.setdefault(
                                fragment["index"],
                                {"id": "", "type": "", "name": "", "arguments": ""},
                            )
                            for key in ("id", "type"):
                                if fragment.get(key):
                                    call[key] = fragment[key]
                            for key in ("name", "arguments"):
                                call[key] += fragment.get("function", {}).get(key) or ""
                        if choice.get("finish_reason"):
                            sample["finish_reason"] = choice["finish_reason"]
        if not done or sample["finish_reason"] is None or sample["ttft_ms"] is None:
            sample["error"] = "incomplete_stream"
        elif sample["finish_reason"] in ("length", "content_filter"):
            sample["error"] = sample["finish_reason"]
        else:
            sample["ok"] = True
    except httpx.HTTPStatusError as exc:
        sample["error"] = f"http_{exc.response.status_code}"
    except (
        httpx.HTTPError,
        TimeoutError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        sample["error"] = type(exc).__name__
    sample["e2e_ms"] = (time.perf_counter() - start) * 1000
    sample["tool_calls"] = len(calls)
    sample["valid_tool_calls"] = sum(valid_call(call) for call in calls.values())
    sample["tool_response_valid"] = bool(
        sample["ok"] and calls and sample["valid_tool_calls"] == len(calls)
    )
    return sample


def percentiles(values: list[float]) -> dict:
    # Linear interpolation, equivalent to NumPy's default percentile method.
    return {
        f"p{p}": float(np.percentile(values, p)) if values else None
        for p in (50, 95, 99)
    }


def summarize(samples: list[dict], elapsed: float, workload: str) -> dict:
    good = [s for s in samples if s["ok"]]
    tokens = [s["output_tokens"] for s in good]
    calls = sum(s["tool_calls"] for s in samples)
    errors = {}
    for sample in samples:
        if sample["error"]:
            errors[sample["error"]] = errors.get(sample["error"], 0) + 1
    return {
        "attempted": len(samples),
        "succeeded": len(good),
        "failed": len(samples) - len(good),
        "errors": errors,
        "duration_s": elapsed,
        "ttft_ms": percentiles([s["ttft_ms"] for s in good]),
        "e2e_ms": percentiles([s["e2e_ms"] for s in good]),
        "requests_per_s": len(good) / elapsed,
        "output_tokens_per_s": sum(tokens) / elapsed
        if tokens and None not in tokens
        else None,
        "output_tokens": sum(tokens) if tokens and None not in tokens else None,
        "usage_coverage": sum(t is not None for t in tokens) / len(good)
        if good
        else None,
        "tool_call_validity_rate": (
            sum(s["tool_response_valid"] for s in samples) / len(samples)
            if workload == "director"
            else None
        ),
        "individual_tool_call_validity_rate": (
            sum(s["valid_tool_calls"] for s in samples) / calls if calls else None
        ),
    }


def metric_values(body: str, model: str) -> dict:
    values = {}
    names = {
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_hits",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
    }
    samples = [
        sample
        for family in text_string_to_metric_families(body)
        for sample in family.samples
        if sample.name in names and math.isfinite(sample.value)
    ]
    labels = {s.labels["model_name"] for s in samples if "model_name" in s.labels}
    # vLLM labels metrics with the first served model name. Requests can use
    # another alias (director/vlm). Only infer that alias for a single model.
    selected = (
        model if model in labels else next(iter(labels)) if len(labels) == 1 else None
    )
    for sample in samples:
        if "model_name" not in sample.labels or sample.labels["model_name"] == selected:
            values.setdefault(sample.name, []).append(sample.value)
    # KV utilization is a gauge per engine; preserve the highest engine value.
    return {
        "values": {
            name: max(v) if "usage" in name else sum(v) for name, v in values.items()
        },
        "matched_model_name": selected,
        "model_match": "exact"
        if selected == model
        else "single_model_alias"
        if selected
        else "unlabelled_or_unmatched",
    }


async def scrape(client: httpx.AsyncClient, url: str, model: str) -> dict:
    try:
        async with asyncio.timeout(5):
            response = await client.get(url, timeout=5)
        response.raise_for_status()
        snapshot = metric_values(response.text, model)
        return {
            "available": bool(snapshot["values"]),
            **snapshot,
            "error": None if snapshot["values"] else "no_matching_metrics",
        }
    except (httpx.HTTPError, ValueError, TimeoutError) as exc:
        return {"available": False, "values": {}, "error": type(exc).__name__}


def metrics_delta(before: dict, after: dict) -> dict:
    def delta(stem):
        for name in (stem + "_total", stem):
            if name in before["values"] and name in after["values"]:
                difference = after["values"][name] - before["values"][name]
                return difference if difference >= 0 else None
        return None

    queries = delta("vllm:prefix_cache_queries")
    hits = delta("vllm:prefix_cache_hits")
    return {
        "before": before,
        "after": after,
        "prefix_cache_queries_delta": queries,
        "prefix_cache_hits_delta": hits,
        "prefix_cache_hit_rate": hits / queries
        if queries and hits is not None and hits <= queries
        else None,
        "kv_usage_note": "before/after snapshots only; not peak utilization",
    }


def endpoint(base: str, suffix: str) -> str:
    parsed = urlsplit(base)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Endpoint must be an http(s) URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(
            "Use API key environment variables, not URL queries or fragments"
        )
    path = parsed.path.rstrip("/")
    if suffix == "metrics" and path.endswith("/v1"):
        path = path[:-3]
    elif suffix != "metrics" and not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/{suffix}", "", ""))


async def run_cell(client, url, bodies, timeout, concurrency):
    samples = []
    queue = iter(enumerate(bodies))

    async def worker():
        for index, body in queue:
            sample = await request(client, url, body, timeout)
            sample["request_index"] = index
            samples.append(sample)

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return samples, time.perf_counter() - start


def model_name(workload: str) -> str:
    return os.environ.get(
        "DIRECTOR_MODEL" if workload == "director" else "ANALYST_MODEL",
        "director" if workload == "director" else "vlm",
    )


async def sweep(args, crop: str) -> list[dict]:
    rows = []
    # Fail before dispatching any traffic if either configured base URL is missing.
    for workload in args.workloads:
        prefix = "DIRECTOR" if workload == "director" else "ANALYST"
        base = os.environ.get(prefix + "_BASE_URL")
        if not base:
            raise ValueError(f"Set {prefix}_BASE_URL before running")
        endpoint(base, "chat/completions")
    for workload in args.workloads:
        prefix = "DIRECTOR" if workload == "director" else "ANALYST"
        base = os.environ.get(prefix + "_BASE_URL")
        url = endpoint(base, "chat/completions")
        metrics_url = os.environ.get(prefix + "_METRICS_URL") or endpoint(
            base, "metrics"
        )
        model = model_name(workload)
        key = os.environ.get(
            prefix + "_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
        # No retries: failures must remain visible in reliability and throughput.
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"},
            timeout=args.timeout,
            limits=httpx.Limits(
                max_connections=max(args.concurrency),
                max_keepalive_connections=max(args.concurrency),
            ),
        ) as client:
            for step, concurrency in enumerate(args.concurrency):
                print(f"{workload}: concurrency={concurrency}", file=sys.stderr)
                warmups = [
                    await request(
                        client,
                        url,
                        payload(
                            workload,
                            1_000_000 + step * args.warmup + i,
                            model,
                            crop,
                            args.max_calls,
                        ),
                        args.timeout,
                    )
                    for i in range(args.warmup)
                ]
                before = (
                    await scrape(client, metrics_url, model) if args.metrics else None
                )
                bodies = [
                    payload(
                        workload, step * args.requests + i, model, crop, args.max_calls
                    )
                    for i in range(args.requests)
                ]
                samples, elapsed = await run_cell(
                    client, url, bodies, args.timeout, concurrency
                )
                after = (
                    await scrape(client, metrics_url, model) if args.metrics else None
                )
                rows.append(
                    {
                        "workload": workload,
                        "model": model,
                        "concurrency": concurrency,
                        "summary": summarize(samples, elapsed, workload),
                        "warmup": {
                            "attempted": len(warmups),
                            "failed": sum(not s["ok"] for s in warmups),
                        },
                        "metrics": metrics_delta(before, after)
                        if args.metrics
                        else None,
                        "samples": sorted(samples, key=lambda s: s["request_index"]),
                    }
                )
    return rows


def compare_runs(baseline: dict, candidate: dict) -> dict:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Baseline schema version differs")
    if baseline.get("prefix_caching") != "off" or candidate["prefix_caching"] != "on":
        raise ValueError("A/B requires an off baseline and an on candidate")
    for key in ("evidence_kind", "workload_fingerprint", "settings", "environment"):
        if baseline.get(key) != candidate[key]:
            raise ValueError(
                f"A/B {key} differs; use identical workload, settings and environment"
            )
    left = {
        (row["workload"], row["model"], row["concurrency"]): row
        for row in baseline["runs"]
    }
    right = {
        (row["workload"], row["model"], row["concurrency"]): row
        for row in candidate["runs"]
    }
    if left.keys() != right.keys():
        raise ValueError("A/B models, workloads or concurrency differ")
    results = []
    for key, row in right.items():
        a, b = left[key]["summary"], row["summary"]

        def change(old, new):
            return (new / old - 1) * 100 if old and new is not None else None

        results.append(
            {
                "workload": key[0],
                "model": key[1],
                "concurrency": key[2],
                "baseline_failed": a["failed"],
                "candidate_failed": b["failed"],
                "ttft_p50_change_pct": change(a["ttft_ms"]["p50"], b["ttft_ms"]["p50"]),
                "e2e_p95_change_pct": change(a["e2e_ms"]["p95"], b["e2e_ms"]["p95"]),
                "output_tokens_per_s_change_pct": change(
                    a["output_tokens_per_s"], b["output_tokens_per_s"]
                ),
                "requests_per_s_change_pct": change(
                    a["requests_per_s"], b["requests_per_s"]
                ),
            }
        )
    return {
        "baseline_created_at": baseline["created_at"],
        "baseline_workload_fingerprint": baseline["workload_fingerprint"],
        "candidate_workload_fingerprint": candidate["workload_fingerprint"],
        "results": results,
        "caveat": "Server settings are operator-declared, not remotely verified. Repeat in reverse order.",
    }


def markdown(result: dict) -> str:
    def fmt(value):
        return "n/a" if value is None else f"{value:.2f}"

    request_counts = ", ".join(str(n) for n in sorted({row["summary"]["attempted"] for row in result["runs"]}))
    lines = [
        "# Sherlock vLLM benchmark",
        "",
        f"Evidence: **{result['evidence_kind']}**. Prefix caching: **{result['prefix_caching']}** (operator-declared).",
        f"Created: {result['created_at']}. Latency in ms; success-only percentiles. No automatic retries.",
        "",
        "| Workload | C | OK / all | TTFT p50 / p95 / p99 | E2E p50 / p95 / p99 | Output tok/s | Req/s | Valid tool responses | Cache hit rate |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    if result["evidence_kind"] == "stub":
        lines.insert(
            3,
            "Stub protocol test only: token counts and cache metrics are synthetic. No model is served.",
        )
    for row in result["runs"]:
        s = row["summary"]
        ttft = " / ".join(fmt(v) for v in s["ttft_ms"].values())
        e2e = " / ".join(fmt(v) for v in s["e2e_ms"].values())
        hit = (row["metrics"] or {}).get("prefix_cache_hit_rate")
        lines.append(
            f"| {row['workload']} | {row['concurrency']} | {s['succeeded']} / {s['attempted']} | {ttft} | {e2e} | {fmt(s['output_tokens_per_s'])} | {fmt(s['requests_per_s'])} | {fmt(s['tool_call_validity_rate'])} | {fmt(hit)} |"
        )
    lines.extend(
        [
            "",
            "Rates are fractions (0–1). Output tok/s = server-reported successful output tokens / sweep wall time; missing usage stays n/a.",
            "TTFT is client-observed time to first text/reasoning/tool-name/argument delta, including network and server queueing.",
            "Tool response validity requires at least one call and every call to pass the imported schema; failed requests count as invalid.",
            "KV usage is a before/after snapshot, not peak. Prefix-cache rates use counter deltas after warmup.",
            f"Requests per cell: {request_counts}. p99 is exploratory with fewer than 200 successful requests per cell.",
        ]
    )
    if result.get("comparison"):
        if result["evidence_kind"] == "stub":
            lines.extend(["", "Stub A/B checks comparison plumbing only; deltas are not model optimization results."])
        lines.extend(
            [
                "",
                "## Prefix caching A/B",
                "",
                "Changes = (on / off - 1) × 100. Negative latency changes are improvements.",
                "",
                "| Workload | C | TTFT p50 % | E2E p95 % | Output tok/s % | Req/s % | Failed off / on |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["comparison"]["results"]:
            lines.append(
                f"| {row['workload']} | {row['concurrency']} | {fmt(row['ttft_p50_change_pct'])} | {fmt(row['e2e_p95_change_pct'])} | {fmt(row['output_tokens_per_s_change_pct'])} | {fmt(row['requests_per_s_change_pct'])} | {row['baseline_failed']} / {row['candidate_failed']} |"
            )
        lines.extend(["", result["comparison"]["caveat"]])
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--workloads",
        nargs="+",
        choices=("director", "vlm"),
        default=["director", "vlm"],
    )
    cli.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    cli.add_argument(
        "--requests",
        type=int,
        default=64,
        help="Measured requests per workload/concurrency cell",
    )
    cli.add_argument(
        "--warmup",
        type=int,
        default=4,
        help="Unmeasured sequential requests before each cell",
    )
    cli.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Per-request wall-clock deadline in seconds",
    )
    cli.add_argument(
        "--max-calls", type=int, default=6, help="Director prompt tool budget"
    )
    cli.add_argument("--image", type=Path, default=FIXTURE)
    cli.add_argument(
        "--metrics", action="store_true", help="Try /metrics before/after each cell"
    )
    cli.add_argument(
        "--prefix-caching",
        choices=("unknown", "off", "on"),
        default="unknown",
        help="Operator declaration; this does not change the server",
    )
    cli.add_argument(
        "--compare",
        type=Path,
        help="Compare against a prior prefix-caching=off JSON file",
    )
    cli.add_argument(
        "--evidence-kind", choices=("endpoint", "stub"), default="endpoint"
    )
    cli.add_argument(
        "--environment",
        default="unspecified",
        help="Nonsecret hardware/model revision/config label, same for A/B",
    )
    cli.add_argument("--stub", action="store_true", help="Run with an owned loopback stub; no GPU or model")
    cli.add_argument("--output-dir", type=Path, help="Default: results/, or .runtime/vllm-stub/ with --stub")
    return cli


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.stub:
        from vllm_stub import run_local

        try:
            return run_local(args, run, ROOT)
        except ValueError as exc:
            cli.error(str(exc))
    args.output_dir = args.output_dir or ROOT / "results"
    return run(args)


def run(args):
    cli = parser()
    if (
        args.requests < 1
        or args.warmup < 0
        or args.max_calls < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or any(c < 1 or c > args.requests for c in args.concurrency)
    ):
        cli.error(
            "Require positive timeout/max-calls, warmup >= 0, and requests >= each positive concurrency"
        )
    args.workloads = list(dict.fromkeys(args.workloads))
    args.concurrency = list(dict.fromkeys(args.concurrency))
    try:
        crop, image_info = load_crop(args.image)
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "evidence_kind": args.evidence_kind,
            "execution_mode": "local_stub" if args.evidence_kind == "stub" else "measured",
            "prefix_caching": args.prefix_caching,
            "environment": args.environment,
            "image": image_info,
            "workload_fingerprint": digest(
                {
                    "script": digest(Path(__file__).read_bytes()),
                    "system": SYSTEM,
                    "weapon_extra_calls": getattr(director, "WEAPON_EXTRA_CALLS", 4),
                    "tools": TOOL_SCHEMAS,
                    "analyst": Analyst.PROMPT,
                    "verifier": Verifier.PROMPT,
                    "image": image_info,
                }
            ),
            "settings": {
                key: getattr(args, key)
                for key in (
                    "workloads",
                    "concurrency",
                    "requests",
                    "warmup",
                    "timeout",
                    "max_calls",
                    "metrics",
                )
            },
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None,
        }
        result["settings"]["models"] = {w: model_name(w) for w in args.workloads}
        # Validate comparison metadata before sending a load test to a remote server.
        baseline = json.loads(args.compare.read_text()) if args.compare else None
        if baseline is not None:
            compare_runs({**baseline, "runs": []}, {**result, "runs": []})
        result["runs"] = asyncio.run(sweep(args, crop))
        if baseline is not None:
            result["comparison"] = compare_runs(baseline, result)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        output = args.output_dir / f"vllm_bench_{stamp}.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        table = markdown(result)
        output.with_suffix(".md").write_text(table)
        print(table)
        print(f"JSON: {output}")
        return (
            1
            if any(
                r["summary"]["failed"] or r["warmup"]["failed"] for r in result["runs"]
            )
            else 0
        )
    except (OSError, ValueError, KeyError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
