"""Ablation switches: every guard and role can be turned off independently."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ARMS: dict[str, dict] = {
    "A0": {},
    "A1": {"filter": False},
    "A2": {"verifier": False},
    "A3": {"reflector": False},
    "A4": {"planner": False},
    "A5": {"memory": False},
    "A6": {"driver": "loop", "filter": False, "verifier": False, "planner": False, "validate_result": False},
    "A7": {"analyzer": False},
    "A9": {"method_card": False},
}


class AblationConfig(BaseModel):
    name: str = "A0"
    planner: bool = True
    filter: bool = True
    matcher: bool = True
    verifier: bool = True
    reflector: bool = True
    analyzer: bool = True
    memory: bool = True
    validate_result: bool = True
    provenance: bool = True
    method_card: bool = True
    driver: Literal["staged", "loop"] = "staged"
    max_reflect_attempts: int = 2

    @classmethod
    def parse(cls, spec: str | dict | None) -> AblationConfig:
        """``"A2"`` | ``"planner=off,verifier=off"`` | dict."""
        if spec is None:
            return cls()
        if isinstance(spec, dict):
            return cls(**spec)
        spec = spec.strip()
        if spec.upper() in ARMS:
            return cls(name=spec.upper(), **ARMS[spec.upper()])
        fields: dict = {}
        for item in spec.split(","):
            if not item.strip():
                continue
            key, _, val = item.partition("=")
            key, val = key.strip(), val.strip().lower()
            if key == "driver":
                fields[key] = val
            elif key == "max_reflect_attempts":
                fields[key] = int(val)
            else:
                fields[key] = val not in ("off", "0", "false", "no")
        return cls(name=spec, **fields)

    def off(self) -> list[str]:
        return [k for k in ("planner", "filter", "matcher", "verifier", "reflector", "analyzer", "memory", "validate_result", "provenance", "method_card") if not getattr(self, k)]


__all__ = ["AblationConfig", "ARMS"]
