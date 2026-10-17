"""Tiny pub/sub so the dashboard sees every stage without coupling to agents."""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    stage: str          # scout | director | tool | analyst | verifier | system
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
        self.stats = {"frames": 0, "scout_hits": 0, "investigations": 0,
                      "vlm_calls": 0, "vlm_calls_avoided": 0, "tool_errors": 0,
                      "recoveries": 0, "confirmed": 0, "rejected": 0,
                      "alerts": 0, "ignored": 0, "logged": 0, "duplicates": 0, "dropped_busy": 0}

    def snapshot(self) -> dict:
        """Stats for display. 'Avoided' = frames the Scout handled that a VLM-per-frame
        system would have sent to a VLM, minus the VLM calls we actually made."""
        s = dict(self.stats)
        s["vlm_calls_avoided"] = max(0, s["frames"] - s["vlm_calls"])
        return s

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def emit(self, stage: str, msg: str, latency_ms: float | None = None, **data) -> None:
        ev = Event(stage, msg, latency_ms, data)
        print(f"[{stage:9}] {msg}" + (f"  ({latency_ms:.0f} ms)" if latency_ms else ""))
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass

    def bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n


BUS = Bus()
