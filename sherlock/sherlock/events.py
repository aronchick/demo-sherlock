"""Tiny pub/sub so the dashboard sees every stage without coupling to agents."""
from __future__ import annotations

import json
import queue
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    stage: str          # scout | director | narration | tool | analyst | verifier | system
    msg: str
    latency_ms: float | None = None
    data: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def json(self) -> str:
        return json.dumps(asdict(self))


class Bus:
    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._capture = threading.local()
        self.stats = {"frames": 0, "scout_hits": 0, "investigations": 0,
                      "vlm_calls": 0, "vlm_calls_avoided": 0, "tool_errors": 0,
                      "recoveries": 0, "confirmed": 0, "rejected": 0,
                      "alerts": 0, "ignored": 0, "logged": 0, "duplicates": 0, "dropped_busy": 0,
                      "evidence_unsaved": 0, "tool_calls": 0}
        # Per-frame detector time for real detectors only; empty in mock mode, so
        # the dashboard never shows a latency nothing measured.
        self.scout_samples: deque[float] = deque(maxlen=300)

    def snapshot(self) -> dict:
        """Stats for display. 'Avoided' = frames the Scout handled that a VLM-per-frame
        system would have sent to a VLM, minus the VLM calls we actually made."""
        s = dict(self.stats)
        s["vlm_calls_avoided"] = max(0, s["frames"] - s["vlm_calls"])
        if self.scout_samples:
            s["scout_ms"] = round(statistics.median(list(self.scout_samples)), 1)
        return s

    def time_scout(self, ms: float) -> None:
        self.scout_samples.append(ms)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    @contextmanager
    def capture(self, events=None):
        """Collect this investigation thread's textual events, never image payloads."""
        previous = getattr(self._capture, "events", None)
        events = [] if events is None else events
        self._capture.events = events
        try:
            yield events
        finally:
            self._capture.events = previous

    def captured_events(self):
        return getattr(self._capture, "events", None)

    def emit(self, stage: str, msg: str, latency_ms: float | None = None, **data) -> None:
        ev = Event(stage, msg, latency_ms, data)
        captured = getattr(self._capture, "events", None)
        if captured is not None and stage != "model_input":
            captured.append({"stage": stage, "t": ev.t, "detail": msg})
        print(f"[{stage:9}] {msg}" + (f"  ({latency_ms:.0f} ms)" if latency_ms else ""), flush=True)  # live in redirected logs
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + n


BUS = Bus()
