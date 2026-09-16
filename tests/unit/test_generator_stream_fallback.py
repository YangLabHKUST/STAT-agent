"""A generation stream that drops mid-way is retried once without streaming; other errors propagate."""

from __future__ import annotations

import asyncio

import pytest

from stat_agent.llm import Completion
from stat_agent.roles.generator import Generation, GeneratorRole, is_transport_error


class RemoteProtocolError(Exception):  # the name is what matters: httpx and its vendored copies share names, not classes
    pass


class _Stub:
    def __init__(self, fail_with: Exception):
        self.fail_with = fail_with
        self.completes = 0

    async def stream_text(self, messages, **kw):
        yield "Counting "
        yield "the cells"
        raise self.fail_with

    async def complete(self, messages, **kw):
        self.completes += 1
        return Completion(text="Recovered.\n```python\nprint('fallback')\n```", mode="text")


async def _run(role):
    items = []
    async for item in role.generate_streaming(GeneratorRole.variables("count the cells", None, None)):
        items.append(item)
    return items


def test_a_dropped_stream_falls_back_to_one_non_streamed_call():
    stub = _Stub(RemoteProtocolError("peer closed connection without sending complete message body"))
    role = GeneratorRole(stub)
    items = asyncio.run(_run(role))
    assert items[:2] == ["Counting ", "the cells"]
    gen = items[-1]
    assert isinstance(gen, Generation) and gen.code_blocks == ["print('fallback')"] and gen.text == "Recovered."
    assert stub.completes == 1


def test_other_stream_errors_propagate():
    stub = _Stub(ValueError("not a transport problem"))
    with pytest.raises(ValueError):
        asyncio.run(_run(GeneratorRole(stub)))
    assert stub.completes == 0


def test_transport_errors_are_recognised_by_name():
    import httpx

    assert is_transport_error(httpx.RemoteProtocolError("x"))
    assert is_transport_error(httpx.ReadTimeout("x"))
    assert is_transport_error(RemoteProtocolError("x"))
    assert not is_transport_error(ValueError("x"))
    assert not is_transport_error(httpx.HTTPStatusError("x", request=None, response=None))
