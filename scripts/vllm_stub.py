"""Loopback-only OpenAI protocol stub. No model inference or GPU metrics."""

from __future__ import annotations

import json
import os
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def schema_example(schema):
    """Build arguments from the request's current tool schema, not copied tools."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {key: schema_example(schema["properties"][key]) for key in schema.get("required", [])}
    if kind == "array":
        return [schema_example(schema["items"]) for _ in range(schema.get("minItems", 1))]
    if kind == "string":
        return "hero"
    if kind in {"integer", "number"}:
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    raise ValueError(f"Unsupported stub schema type: {kind}")


@contextmanager
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_args):
            pass

        def send(self, data, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data.encode())

        def do_GET(self):
            if self.path == "/v1/models":
                self.send(json.dumps({"data": [{"id": model} for model in ("director", "vlm")]}))
            elif self.path == "/metrics":
                with self.server.lock:
                    requests = self.server.requests
                self.send("\n".join(
                    f'{name}{{model_name="{model}"}} {value}'
                    for model in ("director", "vlm")
                    for name, value in (
                        ("vllm:prefix_cache_queries_total", requests * 100),
                        ("vllm:prefix_cache_hits_total", requests * 50),
                        ("vllm:kv_cache_usage_perc", 0.25),
                    )
                ) + "\n", "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.lock:
                self.server.requests += 1
            message = {"role": "assistant", "content": "A white model spacecraft with dark windows."}
            finish = "stop"
            if body.get("tools"):
                tool = body["tools"][0]["function"]
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_stub", "type": "function", "function": {
                        "name": tool["name"], "arguments": json.dumps(schema_example(tool["parameters"]))}}]}
                finish = "tool_calls"
            usage = {"prompt_tokens": 100, "completion_tokens": 10}
            try:
                if body.get("stream"):
                    for index, call in enumerate(message.get("tool_calls", [])):
                        call["index"] = index
                    events = [
                        {"choices": [{"delta": message}]},
                        {"choices": [{"delta": {}, "finish_reason": finish}]},
                        {"choices": [], "usage": usage},
                    ]
                    self.send("".join("data: " + json.dumps(event) + "\n\n" for event in events)
                              + "data: [DONE]\n\n", "text/event-stream")
                else:
                    self.send(json.dumps({"choices": [{"message": message, "finish_reason": finish}],
                                          "usage": usage}))
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        request_queue_size = 128

    server = Server(("127.0.0.1", 0), Handler)
    server.requests, server.lock = 0, threading.Lock()
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    print(f"STUB ONLY: no model is served. Loopback port {port}.", flush=True)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
        print(f"Stub stopped; port {port} is free", flush=True)


def run_local(args, run, root):
    args.output_dir = (args.output_dir or root / ".runtime" / "vllm-stub").resolve()
    if args.output_dir.is_relative_to((root / "results").resolve()):
        raise ValueError("Stub output must be outside results/ (including symlinks)")
    args.evidence_kind = "stub"
    args.environment = "local-stub-no-model"
    with endpoint() as base:
        env = {"OPENAI_API_KEY": "stub-only"}
        for prefix, model in (("DIRECTOR", "director"), ("ANALYST", "vlm"), ("VERIFIER", "vlm")):
            env.update({f"{prefix}_BASE_URL": base, f"{prefix}_MODEL": model, f"{prefix}_API_KEY": "stub-only"})
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            return run(args)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
