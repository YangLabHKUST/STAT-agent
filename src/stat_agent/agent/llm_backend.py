"""STAT 0.1 compatibility for skill code that calls a model itself.

Vendored skill code written for 0.1 does ``from stat_agent.agent.llm_backend import LLMBackend`` and
``await LLMBackend(system_prompt=..., model=..., api_key=..., endpoint=...).run(prompt)``, taking its
arguments from ``session.llm_config`` (which the agent fills from its own model spec). This keeps that
surface on the 2.0 client: the model id is resolved like everywhere else (``provider/model``, keys from
the environment when none is given) and ``endpoint`` is the base URL.
"""

from __future__ import annotations

from typing import Any

from stat_agent.llm.client import LLMClient
from stat_agent.llm.models import make_spec


class LLMBackend:
    def __init__(
        self,
        system_prompt: str = "",
        model: str = "",
        api_key: str | None = None,
        endpoint: str | None = None,
        provider: str | None = None,
        client: LLMClient | None = None,
        **_: Any,
    ) -> None:
        if client is None and not model:
            raise ValueError("LLMBackend needs a model id such as 'poe/claude-opus-4.8' (see session.llm_config)")
        self.system_prompt = system_prompt
        self.client = client or LLMClient(spec=make_spec(model, provider=provider, api_key=api_key, base_url=endpoint))

    async def run(self, prompt: str, **_: Any) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return (await self.client.complete(messages, role="skill")).text


__all__ = ["LLMBackend"]
