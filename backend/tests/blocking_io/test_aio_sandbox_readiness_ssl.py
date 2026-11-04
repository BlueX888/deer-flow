"""The async AIO readiness probe must not load TLS trust on the event loop."""

import ssl
import threading

import httpx
import pytest

from deerflow.community.aio_sandbox.backend import wait_for_sandbox_ready_async


@pytest.mark.asyncio
async def test_aio_sandbox_readiness_tls_setup_runs_off_loop(monkeypatch):
    loop_thread = threading.get_ident()
    original_context = ssl.create_default_context
    contexts = []
    clients = []

    def create_context(*args, **kwargs):
        assert threading.get_ident() != loop_thread, "TLS trust loading blocks the event loop"
        context = original_context(*args, **kwargs)
        contexts.append(context)
        return context

    async def send(self, request, **kwargs):
        assert threading.get_ident() == loop_thread
        clients.append(self)
        return httpx.Response(200, request=request, json={"status": "ok"})

    monkeypatch.setattr(ssl, "create_default_context", create_context)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    assert await wait_for_sandbox_ready_async("https://sandbox.invalid", timeout=5, poll_interval=0.01) is True
    assert contexts
    assert len(clients) == 1
    assert clients[0].is_closed
