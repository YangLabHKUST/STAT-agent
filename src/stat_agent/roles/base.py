"""Role base class: a prompt, a schema, and one LLM call with logging."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from stat_agent.llm import Completion, LLMClient
from stat_agent.prompts import Prompt, load_prompt


class Role:
    name: ClassVar[str] = "role"
    prompt_name: ClassVar[str] = ""
    system_prompt_name: ClassVar[str | None] = None

    def __init__(self, llm: LLMClient, prompt_version: int | None = None) -> None:
        self.llm = llm
        self.prompt: Prompt = load_prompt(self.prompt_name, prompt_version)
        self.system_prompt: Prompt | None = load_prompt(self.system_prompt_name) if self.system_prompt_name else None

    @property
    def prompt_id(self) -> str:
        return self.prompt.prompt_id

    def messages(self, variables: dict[str, Any]) -> list[tuple[str, str]]:
        user = self.prompt.render(**variables)
        msgs: list[tuple[str, str]] = []
        if self.system_prompt is not None:
            msgs.append(("system", self.system_prompt.render()))
        msgs.append(("user", user))
        return msgs

    async def call(
        self,
        variables: dict[str, Any],
        *,
        schema: type[BaseModel] | None = None,
        fake_input: Any = None,
        turn_id: str | None = None,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Completion:
        return await self.llm.complete(
            self.messages(variables),
            role=self.name,
            schema=schema,
            prompt_id=self.prompt_id,
            fake_input=fake_input if fake_input is not None else variables,
            turn_id=turn_id,
            step=step,
            metadata=metadata,
        )


__all__ = ["Role"]
