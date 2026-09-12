"""Typed events: catalogue, safe encoding and emitters."""

from stat_agent.events import catalogue
from stat_agent.events.catalogue import EventBase, parse_event, to_wire
from stat_agent.events.emit import (
    CallbackEmitter,
    Emitter,
    JsonlSink,
    ListEmitter,
    MultiEmitter,
    StampingEmitter,
)
from stat_agent.events.encoding import dumps, to_jsonable

__all__ = [
    "catalogue",
    "EventBase",
    "parse_event",
    "to_wire",
    "Emitter",
    "ListEmitter",
    "CallbackEmitter",
    "JsonlSink",
    "MultiEmitter",
    "StampingEmitter",
    "dumps",
    "to_jsonable",
]
