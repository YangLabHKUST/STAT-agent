"""LLMClient: one entry point for every role call.

* ``complete(messages, role=..., schema=Model)`` returns a validated pydantic object;
* prompted JSON is the default structured mode for every provider (the role prompts
  already specify their JSON format), so all backbones see identical prompts; native
  modes (OpenAI json_schema, tool-forced Anthropic, Gemini schema) are opt-in;
* one repair call on validation failure, then :class:`StructuredOutputError`;
* fixture replay/record for tests and benchmarks (``FixtureStore``);
* every call is written through the :class:`CallLogger` and counted by the ``UsageMeter``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from stat_agent.llm.calllog import CallLogger, CallRecord
from stat_agent.llm.fake import FixtureStore, fixture_key
from stat_agent.llm.models import ModelSpec, make_model, make_spec
from stat_agent.llm.structured import (
    DEFAULT_CAPABILITIES,
    StructuredOutputError,
    native_method_for,
    parse_json_to,
    repair_instruction,
    schema_instruction,
    usage_of,
)
from stat_agent.llm.usage import CallUsage, UsageMeter

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MessageLike = tuple[str, str] | dict[str, str] | Any


def to_lc_messages(messages: Sequence[MessageLike]) -> list[Any]:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

    out: list[Any] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if isinstance(m, dict):
            role, content = m.get("role", "user"), m.get("content", "")
        else:
            role, content = m
        role = role.lower()
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text") and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


PROVIDER_ERROR_NAMES = ("APIConnectionError", "OpenAIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "ServiceUnavailableError", "ConnectError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError", "ConnectionError", "TimeoutError", "IncompleteRead", "ChunkedEncodingError", "StreamChunkTimeoutError")
PROVIDER_ERROR_TEXT = ("connection error", "connection reset", "connection aborted", "timed out", "temporarily unavailable", "overloaded", "rate limit", "bad gateway", "gateway time", "server error", "empty response", "eof occurred", "error code: 429", "error code: 5", "error code: 408", "no streaming chunk received")


def is_provider_error(message: str, code: str | None = None) -> bool:
    """A network or provider-side failure (worth retrying), not a failure of the request or of STAT."""
    if code and code in PROVIDER_ERROR_NAMES:
        return True
    msg = (message or "").lower()
    return any(n.lower() in msg for n in PROVIDER_ERROR_NAMES) or any(t in msg for t in PROVIDER_ERROR_TEXT)


def is_provider_exception(exc: BaseException) -> bool:
    """``is_provider_error`` for an exception: its message or any class it inherits from names a provider failure
    (``langchain_openai`` re-raises ``openai.APIConnectionError`` as its own ``OpenAIConnectionError``)."""
    return any(is_provider_error(str(exc), c.__name__) for c in type(exc).__mro__)


def messages_to_records(messages: Sequence[Any]) -> list[dict[str, str]]:
    recs = []
    for m in messages:
        role = getattr(m, "type", None) or "user"
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        recs.append({"role": role, "content": content_text(m)})
    return recs


@dataclass
class Completion:
    text: str
    value: Any = None
    mode: str = "text"
    repairs: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    fixture_key: str | None = None
    message: Any = None

    @property
    def usage(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


class LLMClient:
    def __init__(
        self,
        model: Any = None,
        *,
        spec: ModelSpec | None = None,
        capabilities: dict[str, str] | None = None,
        structured_default: str | None = None,
        calllog: CallLogger | None = None,
        meter: UsageMeter | None = None,
        fixtures: FixtureStore | None = None,
        emitter: Any = None,
        model_name: str | None = None,
        responder: Any = None,
    ) -> None:
        if isinstance(model, str):
            spec = spec or make_spec(model)
            model = None
        self.spec = spec
        self._model = model
        self.provider = spec.provider if spec else "unknown"
        self.model_name = model_name or (spec.model if spec else getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__ if model is not None else "fake")
        self.capabilities = dict(DEFAULT_CAPABILITIES)
        if capabilities:
            self.capabilities.update(capabilities)
        # Default structured mode: prompted everywhere unless explicitly configured.
        self.structured_default = structured_default or os.environ.get("STAT_STRUCTURED_MODE", "prompted")
        self.calllog = calllog or CallLogger()
        self.meter = meter or UsageMeter()
        self.fixtures = fixtures
        self.emitter = emitter
        # Optional scripted backend: responder(role, messages, schema) -> str (tests, vanilla arms).
        self.responder = responder

    # ---- model ------------------------------------------------------------------------------------
    @property
    def model(self) -> Any:
        if self._model is None:
            if self.spec is None:
                raise RuntimeError("LLMClient has no model (fixture-only client used outside replay)")
            self._model = make_model(self.spec)
        return self._model

    def structured_mode(self, override: str | None = None) -> str:
        if override:
            return override
        if self.structured_default == "native":
            return self.capabilities.get(self.provider, "prompted")
        return self.structured_default

    # ---- main entry points ----------------------------------------------------------------------
    async def complete(
        self,
        messages: Sequence[MessageLike],
        *,
        role: str,
        schema: type[T] | None = None,
        prompt_id: str | None = None,
        fake_input: Any = None,
        turn_id: str | None = None,
        step: int | None = None,
        mode: str | None = None,
        append_schema: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Completion:
        lc_messages = to_lc_messages(messages)
        key = None
        if self.fixtures is not None and self.fixtures.mode != "off":
            key = fixture_key(role, prompt_id, fake_input if fake_input is not None else messages_to_records(lc_messages))
            rec = self.fixtures.lookup(role, key)
            if rec is not None:
                text = rec["response_text"]
                value = parse_json_to(schema, text) if schema is not None else text
                comp = Completion(text=text, value=value, mode="replay", input_tokens=int(rec.get("input_tokens", 0)), output_tokens=int(rec.get("output_tokens", 0)), fixture_key=key)
                self._record(role, prompt_id, lc_messages, comp, turn_id, step, metadata)
                return comp
            if self.fixtures.mode in ("replay", "strict"):
                raise self.fixtures.miss(role, key, key)

        start = time.perf_counter()
        error: str | None = None
        try:
            if self.responder is not None:
                comp = await self._call_scripted(lc_messages, role, schema)
            elif schema is None:
                comp = await self._call_text(lc_messages)
            else:
                comp = await self._call_structured(lc_messages, schema, self.structured_mode(mode), append_schema)
        except StructuredOutputError as exc:
            error = str(exc)
            comp = Completion(text=exc.raw or "", value=None, mode="failed")
            comp.latency_s = time.perf_counter() - start
            self._record(role, prompt_id, lc_messages, comp, turn_id, step, metadata, error=error)
            raise
        comp.latency_s = time.perf_counter() - start
        comp.fixture_key = key
        self._record(role, prompt_id, lc_messages, comp, turn_id, step, metadata)
        if self.fixtures is not None and self.fixtures.mode == "record" and key is not None:
            self.fixtures.store(
                role,
                key,
                {
                    "role": role,
                    "prompt_id": prompt_id,
                    "model": self.model_name,
                    "response_text": comp.text,
                    "input_tokens": comp.input_tokens,
                    "output_tokens": comp.output_tokens,
                    "mode": comp.mode,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
        return comp

    def complete_sync(self, *args: Any, **kwargs: Any) -> Completion:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.complete(*args, **kwargs))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(self.complete(*args, **kwargs))).result()

    async def stream_text(
        self,
        messages: Sequence[MessageLike],
        *,
        role: str,
        prompt_id: str | None = None,
        turn_id: str | None = None,
        step: int | None = None,
        fake_input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas; the full completion is logged when the stream ends."""
        lc_messages = to_lc_messages(messages)
        if self.responder is not None or (self.fixtures is not None and self.fixtures.mode != "off"):
            comp = await self.complete(messages, role=role, prompt_id=prompt_id, fake_input=fake_input, turn_id=turn_id, step=step, metadata=metadata)
            yield comp.text
            return
        start = time.perf_counter()
        pieces: list[str] = []
        in_tok = out_tok = 0
        async for chunk in self.model.astream(lc_messages):
            delta = content_text(chunk)
            um = getattr(chunk, "usage_metadata", None)
            if um:
                in_tok = max(in_tok, int(um.get("input_tokens") or 0))
                out_tok = max(out_tok, int(um.get("output_tokens") or 0))
            if delta:
                pieces.append(delta)
                yield delta
        comp = Completion(text="".join(pieces), value="".join(pieces), mode="stream", input_tokens=in_tok, output_tokens=out_tok, latency_s=time.perf_counter() - start)
        self._record(role, prompt_id, lc_messages, comp, turn_id, step, metadata)

    # ---- internals ---------------------------------------------------------------------------------------
    async def _call_scripted(self, lc_messages: list[Any], role: str, schema: type[T] | None) -> Completion:
        import inspect

        out = self.responder(role, lc_messages, schema)
        if inspect.isawaitable(out):
            out = await out
        text = out if isinstance(out, str) else (out.model_dump_json() if hasattr(out, "model_dump_json") else str(out))
        value = parse_json_to(schema, text) if schema is not None else text
        n_in = len(" ".join(content_text(m) for m in lc_messages)) // 4
        return Completion(text=text, value=value, mode="scripted", input_tokens=n_in, output_tokens=len(text) // 4)

    async def _call_text(self, lc_messages: list[Any]) -> Completion:
        msg = await self.model.ainvoke(lc_messages)
        text = content_text(msg)
        i, o = usage_of(msg)
        return Completion(text=text, value=text, mode="text", input_tokens=i, output_tokens=o, message=msg)

    async def _call_structured(self, lc_messages: list[Any], schema: type[T], mode: str, append_schema: bool) -> Completion:
        from langchain_core.messages import AIMessage, HumanMessage

        in_tok = out_tok = 0
        if mode != "prompted":
            method = native_method_for(self.provider, mode)
            try:
                kwargs = {"include_raw": True}
                if method:
                    kwargs["method"] = method
                runnable = self.model.with_structured_output(schema, **kwargs)
                out = await runnable.ainvoke(lc_messages)
                raw, parsed, perr = out.get("raw"), out.get("parsed"), out.get("parsing_error")
                i, o = usage_of(raw)
                in_tok, out_tok = in_tok + i, out_tok + o
                if parsed is not None:
                    return Completion(text=content_text(raw), value=parsed, mode=mode, input_tokens=in_tok, output_tokens=out_tok, message=raw)
                logger.warning("native structured output failed (%s); falling back to prompted JSON", perr)
            except StructuredOutputError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider/proxy quirks must not fail the turn
                logger.warning("native structured output unavailable (%s: %s); falling back to prompted JSON", type(exc).__name__, exc)

        messages = list(lc_messages)
        if append_schema:
            last = messages[-1]
            messages[-1] = HumanMessage(content=content_text(last) + schema_instruction(schema))
        msg = await self.model.ainvoke(messages)
        text = content_text(msg)
        i, o = usage_of(msg)
        in_tok, out_tok = in_tok + i, out_tok + o
        try:
            value = parse_json_to(schema, text)
            return Completion(text=text, value=value, mode="prompted", input_tokens=in_tok, output_tokens=out_tok, message=msg)
        except StructuredOutputError as first:
            repair_msgs = messages + [AIMessage(content=text), HumanMessage(content=repair_instruction(schema, text, first))]
            msg2 = await self.model.ainvoke(repair_msgs)
            text2 = content_text(msg2)
            i, o = usage_of(msg2)
            in_tok, out_tok = in_tok + i, out_tok + o
            try:
                value = parse_json_to(schema, text2)
            except StructuredOutputError as second:
                raise StructuredOutputError(f"structured output failed after repair: {second}", raw=text2, errors=second.errors) from second
            return Completion(text=text2, value=value, mode="prompted+repair", repairs=1, input_tokens=in_tok, output_tokens=out_tok, message=msg2)

    def _record(self, role: str, prompt_id: str | None, lc_messages: list[Any], comp: Completion, turn_id: str | None, step: int | None, metadata: dict[str, Any] | None, error: str | None = None) -> None:
        usage = CallUsage(role=role, model=self.model_name, input_tokens=comp.input_tokens, output_tokens=comp.output_tokens, latency_s=comp.latency_s, turn_id=turn_id, step=step, mode=comp.mode, repairs=comp.repairs)
        self.meter.record(usage)
        rec = CallRecord(
            role=role,
            model=self.model_name,
            prompt_id=prompt_id,
            messages=messages_to_records(lc_messages),
            response_text=comp.text,
            parsed=comp.value.model_dump(mode="json") if isinstance(comp.value, BaseModel) else (comp.value if not isinstance(comp.value, str) else None),
            mode=comp.mode,
            repairs=comp.repairs,
            input_tokens=comp.input_tokens,
            output_tokens=comp.output_tokens,
            latency_s=comp.latency_s,
            turn_id=turn_id,
            step=step,
            error=error,
            metadata=dict(metadata or {}),
        )
        try:
            self.calllog.record(rec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("call log write failed: %s", exc)
        if self.emitter is not None:
            try:
                from stat_agent.events.catalogue import UsageEvent

                totals = self.meter.turn_totals(turn_id) if turn_id else self.meter.total
                self.emitter.emit(UsageEvent(role=role, llm_calls=totals.llm_calls, tokens_in=totals.input_tokens, tokens_out=totals.output_tokens, elapsed_ms=int(totals.latency_s * 1000), model=self.model_name, turn_id=turn_id, step=step))
            except Exception as exc:  # noqa: BLE001
                logger.debug("usage event emit failed: %s", exc)


__all__ = ["LLMClient", "Completion", "to_lc_messages", "content_text", "messages_to_records", "is_provider_error", "is_provider_exception"]
