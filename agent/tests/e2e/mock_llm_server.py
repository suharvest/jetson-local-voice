"""Pluggable mock LLM server with OpenAI-compatible endpoints.

Used by ``test_llm_resilience.py`` to exercise the agent's LLM
failure-handling without depending on a real upstream.
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any, Awaitable, Callable

from aiohttp import web

ScenarioFn = Callable[
    ["MockLLMServer", web.Request, dict[str, Any]],
    Awaitable[web.StreamResponse],
]


def _sse_chunk(
    token: str,
    *,
    finish_reason: str | None = None,
    model: str = "mock",
) -> bytes:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": token} if token else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def _non_stream_response(tokens: list[str]) -> web.Response:
    content = "".join(tokens)
    body = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": len(tokens),
                  "total_tokens": 1 + len(tokens)},
    }
    return web.json_response(body)


def _is_probe_body(body: dict) -> bool:
    """Heuristic: the LLMAvailability plugin sends ``stream=False`` AND
    ``max_tokens<=4``. Real user turns always stream."""
    return (
        body.get("stream") is False
        and isinstance(body.get("max_tokens"), int)
        and body["max_tokens"] <= 4
    )


async def _stream_tokens(
    request: web.Request,
    tokens: list[str],
    *,
    per_token_delay_s: float = 0.0,
    finish_reason_last: str = "stop",
) -> web.StreamResponse:
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
    )
    await resp.prepare(request)
    for tok in tokens:
        await resp.write(_sse_chunk(tok))
        if per_token_delay_s:
            await asyncio.sleep(per_token_delay_s)
    await resp.write(_sse_chunk("", finish_reason=finish_reason_last))
    await resp.write(_sse_done())
    await resp.write_eof()
    return resp


class MockLLMServer:
    """Tiny aiohttp server exposing ``/v1/chat/completions`` and ``/v1/models``."""

    def __init__(self) -> None:
        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", self._handle_chat)
        self.app.router.add_get("/v1/models", self._handle_models)
        self.scenarios: list[ScenarioFn] = []
        # Every inbound /v1/chat/completions request (probe + real turn).
        self.requests_received: list[dict[str, Any]] = []
        # Only request bodies for *real* (streaming) turns. Tests assert
        # against this list so probe traffic doesn't shift indices.
        self.chat_requests: list[dict[str, Any]] = []
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.port: int | None = None
        self.models_endpoint_always_ok: bool = True
        # If True, probes always fail with HTTP 500 (used by scenarios 1
        # and 4 where the test wants the availability machine to converge
        # to DOWN). If False, probes get a baked "ok" reply that doesn't
        # consume scenarios queued for real turns.
        self.probes_always_500: bool = False

    async def start(self, host: str = "127.0.0.1") -> str:
        sock = socket.socket()
        sock.bind((host, 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host, self.port)
        await self.site.start()
        return f"http://{host}:{self.port}"

    async def stop(self) -> None:
        if self.site is not None:
            try:
                await self.site.stop()
            except Exception:
                pass
            self.site = None
        if self.runner is not None:
            try:
                await self.runner.cleanup()
            except Exception:
                pass
            self.runner = None

    async def _handle_models(self, request: web.Request) -> web.Response:
        if self.models_endpoint_always_ok:
            return web.json_response(
                {"object": "list", "data": [{"id": "mock", "object": "model"}]}
            )
        return web.json_response({"error": "models down"}, status=503)

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        self.requests_received.append(body)
        is_probe = _is_probe_body(body)
        # Probes are served by a dedicated path so they don't consume
        # scenarios queued for real user turns.
        if is_probe:
            if self.probes_always_500:
                return web.json_response(
                    {"error": "mock probe down"}, status=500,
                )
            return _non_stream_response(["."])
        self.chat_requests.append(body)
        scenario = self.scenarios.pop(0) if self.scenarios else _default_success
        return await scenario(self, request, body)

    def enqueue(self, scenario: ScenarioFn) -> None:
        self.scenarios.append(scenario)

    def enqueue_success(
        self,
        tokens: list[str] | None = None,
        per_token_delay_s: float = 0.0,
    ) -> None:
        toks = list(tokens) if tokens is not None else ["你", "好", "啊"]

        async def _scenario(server, request, body):  # noqa: ANN001
            return await _stream_tokens(
                request,
                toks,
                per_token_delay_s=per_token_delay_s,
            )

        self.enqueue(_scenario)

    def enqueue_500(self, message: str = "mock server error") -> None:
        async def _scenario(server, request, body):  # noqa: ANN001
            return web.json_response({"error": message}, status=500)

        self.enqueue(_scenario)

    def enqueue_502(self, message: str = "mock bad gateway") -> None:
        async def _scenario(server, request, body):  # noqa: ANN001
            return web.json_response({"error": message}, status=502)

        self.enqueue(_scenario)

    def enqueue_finish_reason_error(self, n_tokens_before: int = 2) -> None:
        async def _scenario(server, request, body):  # noqa: ANN001
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await resp.prepare(request)
            for i in range(n_tokens_before):
                await resp.write(_sse_chunk(f"t{i}"))
            await resp.write(_sse_chunk("", finish_reason="error"))
            await resp.write(_sse_done())
            await resp.write_eof()
            return resp

        self.enqueue(_scenario)

    def enqueue_400_prefix_cache(self) -> None:
        async def _scenario(server, request, body):  # noqa: ANN001
            extra = body.get("extra_body") or {}
            uses_pc = bool(body.get("prefix_cache") or extra.get("prefix_cache"))
            if uses_pc:
                return web.json_response(
                    {"error": "prefix_cache miss: prefix_messages required"},
                    status=400,
                )
            return await _stream_tokens(request, ["o", "k"])

        self.enqueue(_scenario)


async def _default_success(
    server: MockLLMServer,
    request: web.Request,
    body: dict[str, Any],
) -> web.StreamResponse:
    return await _stream_tokens(request, ["o", "k"])


__all__ = ["MockLLMServer"]
