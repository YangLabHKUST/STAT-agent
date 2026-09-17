"""STAT 0.1 compatibility for skill code that calls a model itself.

Vendored skill code written for 0.1 does ``from stat_agent.agent.llm_backend import LLMBackend`` and
``await LLMBackend(system_prompt=..., model=..., api_key=..., endpoint=...).run(prompt)``, taking its
arguments from ``session.llm_config`` (which the agent fills from its own model spec). This keeps that
surface on the 2.0 client: the model id is resolved like everywhere else (``provider/model``, keys from
the environment when none is given) and ``endpoint`` is the base URL. A network or provider failure is
retried after each of ``RETRY_DELAYS_S`` before it reaches the skill.
"""

from __future__ import annotations

import asyncio
from typing import Any

from stat_agent.llm.client import LLMClient, is_provider_exception
from stat_agent.llm.models import make_spec

RETRY_DELAYS_S = (5, 15, 45, 120, 300)


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
        for retry, delay in enumerate(RETRY_DELAYS_S, start=1):
            try:
                return (await self.client.complete(messages, role="skill")).text
            except Exception as exc:  # noqa: BLE001 - only a provider failure is retried; anything else reaches the skill
                if not is_provider_exception(exc):
                    raise
                # printed: skill code runs in the executor, where stdout is what the user sees while it waits
                print(f"Model call failed ({type(exc).__name__}: {exc}); retry {retry} of {len(RETRY_DELAYS_S)} in {delay} s", flush=True)
                await asyncio.sleep(delay)
        return (await self.client.complete(messages, role="skill")).text


__all__ = ["LLMBackend", "RETRY_DELAYS_S"]
