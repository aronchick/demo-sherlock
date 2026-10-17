# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27,<1", "jsonschema>=4.23,<5", "numpy>=1.26",
#   "opencv-python>=4.10", "openai>=1.40", "python-dotenv>=1",
# ]
# ///
"""Check existing vLLM servers or run a local Brev prefix-cache A/B.

Run on the GPU host after cloning the repo; this never provisions a machine.
The ab mode restarts director/vlm and leaves the cache-on candidate serving.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import httpx
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sherlock"))
BENCH = ROOT / "scripts/bench_vllm.py"
IMAGE = ROOT / "tests/fixtures/vllm/shuttle_crop.jpg"
VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"


def api_url(base, resource):
    url = urlsplit(base)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Use an http(s) base URL without credentials/query/fragment")
    path = url.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((url.scheme, url.netloc, f"{path}/{resource}", "", ""))


def endpoints():
    return {
        role: {
            "base": os.environ.get(f"{prefix}_BASE_URL", f"http://127.0.0.1:{port}/v1"),
            "model": os.environ.get(f"{prefix}_MODEL"),
            "key": os.environ.get(
                f"{prefix}_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")
            ),
        }
        for role, prefix, port in (
            ("director", "DIRECTOR", 8001),
            ("vlm", "ANALYST", 8002),
        )
    }


async def wait_models(role, ep, timeout, poll):
    deadline = time.monotonic() + timeout
    last = "not reachable"
    url = api_url(ep["base"], "models")
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {ep['key']}"}
    ) as client:
        while time.monotonic() < deadline:
            try:
                async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
                    response = await client.get(url, timeout=min(10, timeout))
                    response.raise_for_status()
                    data = response.json()["data"]
                    if not isinstance(data, list):
                        raise TypeError("invalid model list")
                    ids = [item["id"] for item in data]
                    if not ids or not all(isinstance(mid, str) and mid for mid in ids):
                        raise ValueError("empty/invalid model list")
                    model = ep["model"] or (
                        role if role in ids else ids[0] if len(ids) == 1 else None
                    )
                    if model not in ids:
                        raise ValueError("requested model alias absent or ambiguous")
                    print(f"{role}: served model ids = {json.dumps(ids)}", flush=True)
                    ep["model"] = model
                    return {"ids": ids, "selected": model}
            except httpx.HTTPStatusError as exc:
                last = f"HTTP {exc.response.status_code}"
                if exc.response.status_code in {401, 403}:
                    raise ValueError(
                        f"{role}: authentication failed ({last})"
                    ) from None
            except (
                httpx.HTTPError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                last = type(exc).__name__
            remaining = deadline - time.monotonic()
            if remaining > 0:
                print(
                    f"{role}: waiting for models ({last}); {remaining:.0f}s left",
                    flush=True,
                )
                await asyncio.sleep(min(poll, remaining))
    raise ValueError(f"{role}: model readiness timed out after {timeout:g}s ({last})")


async def ready(eps, args):
    roles = list(eps)
    values = await asyncio.gather(
        *(
            wait_models(role, eps[role], args.ready_timeout, args.poll_interval)
            for role in roles
        )
    )
    return dict(zip(roles, values, strict=True))


def smoke_payloads(eps, image):
    # Import the installed application prompts/schemas on each invocation.
    from sherlock.agents import director
    from sherlock.agents.vision import Analyst
    from sherlock.tools import TOOL_SCHEMAS

    crop = cv2.imread(str(image))
    if crop is None or not image.read_bytes().startswith(b"\xff\xd8"):
        raise ValueError("--image must be a decodable JPEG object crop")
    height, width = crop.shape[:2]
    scale = min(1, 768 / max(height, width))
    crop = cv2.resize(crop, (int(width * scale), int(height * scale)))
    ok, wire = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("JPEG encoding failed")
    event = {
        "camera": "hero",
        "label": "knife",
        "threat": "weapon",
        "confidence": 0.94,
        "bbox": [120, 180, 280, 320],
        "head_bbox": [120, 20, 200, 100],
        "cameras": [{"name": "hero", "pan": 0, "tilt": 0, "zoom": 1, "covered": False}],
    }
    return {
        "director": {
            "model": eps["director"]["model"],
            "temperature": 0.1,
            "max_tokens": 300,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": director.SYSTEM.format(
                        max_calls=6 + getattr(director, "WEAPON_EXTRA_CALLS", 4)
                    ),
                },
                {"role": "user", "content": json.dumps(event)},
            ],
        },
        "vlm": {
            "model": eps["vlm"]["model"],
            "temperature": 0.1,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": Analyst.PROMPT.format(
                                question="What is this object? Read any visible text."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(wire).decode()
                            },
                        },
                    ],
                }
            ],
        },
    }


def check_director(choice, schemas):
    message = choice["message"]
    calls = message.get("tool_calls")
    if (
        choice.get("finish_reason") != "tool_calls"
        or not isinstance(calls, list)
        or not calls
    ):
        raise ValueError("Director smoke: expected a complete tool-call response")
    if (
        message.get("reasoning_content")
        or message.get("reasoning")
        or "<think>" in (message.get("content") or "")
    ):
        raise ValueError("Director smoke: reasoning was returned despite thinking off")
    parameters = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in schemas
    }
    for call in calls:
        function = call["function"]
        schema = parameters.get(function["name"])
        if (
            not isinstance(call.get("id"), str)
            or not call["id"]
            or call.get("type") != "function"
            or schema is None
        ):
            raise ValueError("Director smoke: unknown/malformed tool call")

        def reject_constant(_value):
            raise ValueError("Director smoke: non-JSON numeric constant")

        arguments = json.loads(function["arguments"], parse_constant=reject_constant)
        if (
            not isinstance(arguments, dict)
            or arguments.keys() - schema["properties"].keys()
            or not Draft202012Validator(schema).is_valid(arguments)
        ):
            raise ValueError("Director smoke: arguments do not match runtime schema")
    return {"valid_tool_calls": len(calls), "thinking": False}


async def smoke(eps, bodies, timeout):
    results = {}
    for role, body in bodies.items():
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {eps[role]['key']}"}
        ) as client:
            async with asyncio.timeout(timeout):
                response = await client.post(
                    api_url(eps[role]["base"], "chat/completions"),
                    json=body,
                    timeout=timeout,
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
        if role == "director":
            results[role] = check_director(choice, body["tools"])
        else:
            content = choice["message"].get("content")
            if (
                choice.get("finish_reason") != "stop"
                or not isinstance(content, str)
                or not content.strip()
            ):
                raise ValueError(
                    "VLM smoke: expected a complete nonempty text response"
                )
            results[role] = {"nonempty_response": True}
        print(f"{role}: smoke passed", flush=True)
    return results


def command(argv, *, env=None, capture=False, timeout=900):
    return subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        capture_output=capture,
        text=True,
        check=True,
        timeout=timeout,
    )


def inspect_container(name):
    raw = command(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{json .Config.Image}}\n{{json .Config.Cmd}}\n{{json .Image}}",
            name,
        ],
        capture=True,
    ).stdout
    image, argv, image_id = map(json.loads, raw.splitlines())
    if image != VLLM_IMAGE:
        raise ValueError(f"Refusing to restart {name}: not the configured vLLM image")
    return {"argv": argv, "image_id": image_id}


def restart(prefix_caching):
    existing = command(
        ["docker", "container", "ls", "--all", "--format", "{{.Names}}"], capture=True
    ).stdout.splitlines()
    names = [name for name in ("director", "vlm") if name in existing]
    for name in names:
        inspect_container(name)
    if names:
        print(
            "Restart sequence: stop then remove director/vlm containers (HF cache is retained)",
            flush=True,
        )
        command(["docker", "container", "stop", "--timeout", "30", *names], timeout=90)
        command(["docker", "container", "rm", *names], timeout=30)
    print(f"Starting PREFIX_CACHING={prefix_caching} bash brev/serve.sh", flush=True)
    command(
        ["bash", str(ROOT / "brev/serve.sh")],
        env={**os.environ, "PREFIX_CACHING": str(prefix_caching)},
    )
    config = {name: inspect_container(name) for name in ("director", "vlm")}
    argv = config["director"]["argv"]
    expected = (
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching"
    )
    opposite = (
        "--no-enable-prefix-caching" if prefix_caching else "--enable-prefix-caching"
    )
    if (
        expected not in argv
        or opposite in argv
        or any(
            "speculat" in arg or arg.startswith("--spec-") or arg == "-sc"
            for server in config.values()
            for arg in server["argv"]
        )
    ):
        raise ValueError(
            "Director startup flags do not match cache/no-speculation configuration"
        )
    # Strip only the optimization knobs; compare everything else across legs.
    normalized = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--prefix-caching-hash-algo":
            skip = True
        elif arg not in {"--enable-prefix-caching", "--no-enable-prefix-caching"}:
            normalized.append(arg)
    config["director"]["argv"] = normalized
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def benchmark(args, eps, label, folder, baseline=None):
    env = dict(os.environ)
    for role, prefix in (("director", "DIRECTOR"), ("vlm", "ANALYST")):
        env.update(
            {
                f"{prefix}_BASE_URL": eps[role]["base"],
                f"{prefix}_MODEL": eps[role]["model"],
                f"{prefix}_API_KEY": eps[role]["key"],
            }
        )
    argv = [
        "uv",
        "run",
        str(BENCH),
        "--metrics",
        "--image",
        str(args.image),
        "--requests",
        str(args.requests),
        "--warmup",
        str(args.warmup),
        "--timeout",
        str(args.timeout),
        "--concurrency",
        *map(str, args.concurrency),
        "--prefix-caching",
        label,
        "--environment",
        args.environment,
        "--evidence-kind",
        args.evidence_kind,
        "--output-dir",
        str(folder),
    ]
    if baseline:
        argv += ["--compare", str(baseline)]
    command(argv, env=env, timeout=args.bench_timeout)
    files = list(folder.glob("vllm_bench_*.json"))
    if len(files) != 1:
        raise ValueError("Benchmark did not produce exactly one results JSON")
    return files[0]


def mac_instructions(args, eps):
    print("\nPaste these lines into the Mac .env:")
    for prefix, role, port in (
        ("DIRECTOR", "director", 8001),
        ("ANALYST", "vlm", 8002),
        ("VERIFIER", "vlm", 8002),
    ):
        print(f"{prefix}_BASE_URL=http://{args.mac_host}:{port}/v1")
        print(f"{prefix}_MODEL={json.dumps(eps[role]['model'])}")
    print("DIRECTOR_THINKING=0")
    # The demo's Verifier operating point for SmolVLM-500M (README, Verifier verdicts); the code default is 0.0.
    print("VERIFIER_MARGIN=-1.0")
    print("SHERLOCK_MOCK=0")
    if all(ep["key"] in {"EMPTY", "ollama", ""} for ep in eps.values()):
        print("OPENAI_API_KEY=EMPTY")
    else:
        print("# Set OPENAI_API_KEY privately; credentials are not printed.")
    instance = args.instance or "<instance-name>"
    print("\nOn the Mac: test /v1/models (uses .env, no generation):")
    print("uv run scripts/brev_preflight.py --mode models --ready-timeout 30")
    print("\nIf venue networking blocks direct ports, use two Mac terminals:")
    print(f"brev port-forward {instance} --port 8001:8001")
    print(f"brev port-forward {instance} --port 8002:8002")
    print("Or use one SSH tunnel after brev refresh:")
    print(
        "ssh -N -o ExitOnForwardFailure=yes \\\n  -L 127.0.0.1:8001:127.0.0.1:8001 \\\n  -L 127.0.0.1:8002:127.0.0.1:8002 "
        + instance
    )
    print(
        "Tunnel .env hosts: use 127.0.0.1 for all three BASE_URL values; keep the tunnel running."
    )


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--no-instructions", action="store_true",
                     help="Suppress standalone Mac setup hints when called by a wrapper")
    cli.add_argument(
        "--mode", choices=("models", "smoke", "sweep", "ab"), default="sweep"
    )
    cli.add_argument("--image", type=Path, default=IMAGE)
    cli.add_argument("--ready-timeout", type=float, default=1800)
    cli.add_argument("--poll-interval", type=float, default=5)
    cli.add_argument("--timeout", type=float, default=120)
    cli.add_argument("--bench-timeout", type=float, default=2700)
    cli.add_argument("--requests", type=int, default=200)
    cli.add_argument("--warmup", type=int, default=4)
    cli.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    cli.add_argument(
        "--prefix-caching", choices=("unknown", "off", "on"), default="unknown"
    )
    cli.add_argument("--environment", default="brev-h100-config-a")
    cli.add_argument(
        "--evidence-kind", choices=("endpoint", "stub"), default="endpoint"
    )
    cli.add_argument(
        "--instance", help="Brev instance/SSH alias; only used in printed Mac commands"
    )
    cli.add_argument(
        "--mac-host",
        default="127.0.0.1",
        help="Host for printed Mac .env; defaults to SSH tunnel",
    )
    cli.add_argument("--stub", action="store_true", help="Use a loopback stub; A/B restarts are simulated, never Docker")
    cli.add_argument("--output-dir", type=Path, help="Default: results/, or .runtime/vllm-stub/ with --stub")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.stub:
        from vllm_stub import run_local

        try:
            return run_local(args, run, ROOT)
        except ValueError as exc:
            cli.error(str(exc))
    load_dotenv(ROOT / ".env")
    args.output_dir = args.output_dir or ROOT / "results"
    return run(args)


def run(args):
    cli = parser()
    if (
        any(
            not math.isfinite(v) or v <= 0
            for v in (
                args.ready_timeout,
                args.poll_interval,
                args.timeout,
                args.bench_timeout,
            )
        )
        or args.requests < 1
        or args.warmup < 0
        or any(c < 1 or c > args.requests for c in args.concurrency)
    ):
        cli.error(
            "Require positive timeouts, requests >= concurrency >= 1, warmup >= 0"
        )
    if args.instance and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", args.instance
    ):
        cli.error("--instance must be a Brev instance name/SSH alias")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", args.mac_host):
        cli.error("--mac-host must be a hostname or IPv4 address")
    eps = endpoints()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = args.output_dir / f"brev_preflight_{stamp}"
    report = {
        "schema_version": 1,
        "evidence_kind": args.evidence_kind,
        "execution_mode": "local_stub" if args.evidence_kind == "stub" else "measured",
        "mode": args.mode,
        "ok": False,
        "stages": [],
    }
    try:
        for ep in eps.values():
            api_url(ep["base"], "models")
        if args.mode in {"sweep", "ab"} and not BENCH.is_file():
            raise ValueError(
                "Benchmark script missing: merge benchmark PR #11 before sweep/ab"
            )
        if args.mode != "models" and not args.image.is_file():
            raise ValueError(
                "Object JPEG missing: supply --image or merge benchmark PR #11"
            )
        if args.mode == "ab" and not args.stub:
            for role, port in (("director", 8001), ("vlm", 8002)):
                url = urlsplit(eps[role]["base"])
                if (
                    url.hostname not in {"localhost", "127.0.0.1", "::1"}
                    or url.port != port
                    or url.path.rstrip("/") not in {"", "/v1"}
                ):
                    raise ValueError(
                        "ab must run on the GPU host against local ports 8001/8002"
                    )
        folder.mkdir(parents=True, exist_ok=False)
        baseline, previous_config = None, None
        labels = ("off", "on") if args.mode == "ab" else (args.prefix_caching,)
        for label in labels:
            stage = {"prefix_caching": label}
            report["stages"].append(stage)
            if args.mode == "ab" and args.stub:
                stage["restart"] = "simulated; no containers or cache optimization"
                print(f"STUB A/B: simulated prefix caching {label}; no Docker restart", flush=True)
            if args.mode == "ab" and not args.stub:
                stage["server_config_sha256"] = restart(int(label == "on"))
                if previous_config and previous_config != stage["server_config_sha256"]:
                    raise ValueError(
                        "Server image/arguments changed beyond prefix caching; A/B refused"
                    )
                previous_config = stage["server_config_sha256"]
            stage["models"] = asyncio.run(ready(eps, args))
            if args.mode != "models":
                stage["smoke"] = asyncio.run(
                    smoke(eps, smoke_payloads(eps, args.image), args.timeout)
                )
            if args.mode in {"sweep", "ab"}:
                result = benchmark(args, eps, label, folder / label, baseline)
                stage["benchmark_json"] = str(result)
                baseline = result
        report["ok"] = True
        if args.stub:
            print("STUB ONLY: readiness, smoke, sweep and comparison plumbing; no serving performance evidence.")
        else:
            if not args.no_instructions:
                mac_instructions(args, eps)
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        TimeoutError,
        httpx.HTTPError,
        subprocess.SubprocessError,
    ) as exc:
        # Never dump responses, headers, endpoint URLs, or credential values.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        report["error"] = message
        print(f"Preflight failed: {message}", file=sys.stderr)
        return 1
    finally:
        if folder.is_dir():
            path = folder / "preflight.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Preflight evidence: {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
