"""Closing an event stream releases its bus subscription."""
import asyncio
from types import SimpleNamespace

from sherlock.dashboard import server
from sherlock.events import Bus


def test_event_stream_disconnect(monkeypatch):
    bus = Bus()
    monkeypatch.setattr(server, "BUS", bus)
    app = server.make_app(SimpleNamespace())
    endpoint = next(route.endpoint for route in app.routes if route.path == "/events")

    async def exercise():
        response = await endpoint()
        stream = response.body_iterator
        assert "event: stats" in await anext(stream)
        bus.emit("system", "connected")
        assert "connected" in await anext(stream)
        await stream.aclose()
        bus.emit("system", "disconnected")
        assert bus._subs == []

    asyncio.run(exercise())
