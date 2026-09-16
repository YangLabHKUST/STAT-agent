"""Generator role: explanation text plus fenced Python code (the only free-text role)."""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, Field

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import generator_session_block, history_block, plan_block, skill_guidance_block
from stat_agent.memory.state import ConversationMemory
from stat_agent.roles.base import Role

logger = logging.getLogger(__name__)

_INTERPRET_RE = re.compile(r"__INTERPRET__:\s*(.+?)(?:\n|$)")
_CODE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\n(.*?)```", re.S)
_PYTHON_TAGS = {"python", "py", "python3", "ipython"}


def _is_python_block(tag: str, body: str) -> bool:
    """A fence tagged python (any case), or an untagged fence whose body parses as Python
    statements; bash/json/prose fences are left in the surrounding text."""
    if tag.lower() in _PYTHON_TAGS:
        return True
    if tag or not body.strip():
        return False
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        return False
    return any(not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Name, ast.Constant))) for n in tree.body)

_TRANSPORT_ERROR_NAMES = frozenset({
    "TransportError", "NetworkError", "ProtocolError", "RemoteProtocolError", "LocalProtocolError", "ReadError", "WriteError",
    "ConnectError", "CloseError", "TimeoutException", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout",
    "APIConnectionError", "APITimeoutError", "IncompleteRead", "ChunkedEncodingError", "ServerDisconnectedError", "ClientPayloadError",
})


def is_transport_error(exc: BaseException) -> bool:
    """A dropped, reset or timed-out connection, whichever HTTP stack raised it.

    Matched by class name along the MRO: httpx, the copies of it vendored inside provider SDKs, and the
    SDKs' own connection errors share names, not classes. HTTP status errors are not transport errors."""
    return any(t.__name__ in _TRANSPORT_ERROR_NAMES for t in type(exc).__mro__)


class Segment(BaseModel):
    kind: Literal["text", "code"]
    content: str


class Generation(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    interpret_instruction: str = ""
    raw: str = ""

    @property
    def code_blocks(self) -> list[str]:
        return [s.content for s in self.segments if s.kind == "code"]

    @property
    def has_code(self) -> bool:
        return any(s.kind == "code" for s in self.segments)

    @property
    def text(self) -> str:
        return "\n\n".join(s.content for s in self.segments if s.kind == "text").strip()


def parse_segments(response: str) -> Generation:
    found = [m.group(1).strip() for m in _INTERPRET_RE.finditer(response)]
    interpret = "; ".join(dict.fromkeys(f for f in found if f))  # the model may write one marker per block
    if found:
        response = _INTERPRET_RE.sub("", response)
    segments: list[Segment] = []
    last = 0
    for cm in _CODE_RE.finditer(response):
        code = cm.group(2).strip()
        if not _is_python_block(cm.group(1), code):
            continue
        text = response[last : cm.start()].strip()
        if text:
            segments.append(Segment(kind="text", content=text))
        segments.append(Segment(kind="code", content=code))
        last = cm.end()
    tail = response[last:].strip()
    if tail:
        segments.append(Segment(kind="text", content=tail))
    return Generation(segments=segments, interpret_instruction=interpret, raw=response)


class GeneratorRole(Role):
    name = "generator"
    prompt_name = "generator_user"
    system_prompt_name = "generator_system"

    @staticmethod
    def variables(
        query: str,
        summary: SessionSummary | None,
        memory: ConversationMemory | None,
        *,
        roi_names: list[str] = (),
        skill_guidance: str = "",
        state_block: str = "",
        plan: tuple[str, int, int, str] | None = None,
        unguarded_note: str = "",
        continuation: str = "",
        turn_steps: str = "",
        namespace: str = "",
    ) -> dict:
        has_data = bool(summary and summary.has_data)
        entity = memory.entity_roi_text(list(roi_names)) if memory else ""
        return {
            "plan_block": plan_block(*plan) if plan else "",
            "session_block": generator_session_block(summary, list(roi_names)) if has_data else "",
            "entity_block": entity,
            "has_data": has_data,
            "state_block": state_block,
            "history_block": history_block(memory, "plain"),
            "skill_guidance": skill_guidance,
            "unguarded_note": unguarded_note,
            "continuation_block": continuation,
            "steps_block": turn_steps,
            "namespace_block": namespace,
            "user_query": query,
        }

    async def generate(self, variables: dict, *, fake_input=None, turn_id: str | None = None, step: int | None = None) -> Generation:
        comp = await self.call(variables, fake_input=fake_input, turn_id=turn_id, step=step)
        return parse_segments(comp.text)

    async def generate_streaming(self, variables: dict, *, fake_input=None, turn_id: str | None = None, step: int | None = None) -> AsyncIterator[str | Generation]:
        """Yield text deltas, then the final ``Generation``.

        A connection that drops inside the stream is retried once without streaming (the request-level
        retries of the provider client apply there); the deltas already yielded are superseded by the
        final generation, which the viewer renders in place of the streamed preview."""
        msgs = self.messages(variables)
        pieces: list[str] = []
        try:
            async for delta in self.llm.stream_text(msgs, role=self.name, prompt_id=self.prompt_id, turn_id=turn_id, step=step, fake_input=fake_input if fake_input is not None else variables):
                pieces.append(delta)
                yield delta
        except Exception as exc:  # noqa: BLE001 - only a transport failure is retried; everything else propagates
            if not is_transport_error(exc):
                raise
            logger.warning("generation stream dropped after %d characters (%s: %s); retrying without streaming", sum(len(p) for p in pieces), type(exc).__name__, exc)
            yield await self.generate(variables, fake_input=fake_input, turn_id=turn_id, step=step)
            return
        yield parse_segments("".join(pieces))

    @staticmethod
    def guidance(name: str, body: str, produces: list[str] = (), max_chars: int = 25_000) -> str:
        if len(body) > max_chars:
            body = body[:max_chars] + "\n\n[... skill instructions truncated ...]"
        text = skill_guidance_block(name, body)
        if produces:
            text += "\n\n**Declared outputs of this skill (write them onto the target slice's AnnData; the step is checked for them):** " + ", ".join(produces)
        return text


__all__ = ["GeneratorRole", "Generation", "Segment", "is_transport_error", "parse_segments"]
