"""Request bodies (pydantic) for the JSON API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SessionCreate(_Body):
    dataset_dir: str | None = None
    session_name: str | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    mode: Literal["interactive", "auto"] | None = None
    session_dir: str | None = None  # load a saved session instead of a dataset dir

    @model_validator(mode="after")
    def _one_source(self):
        if not self.dataset_dir and not self.session_dir:
            raise ValueError("dataset_dir or session_dir is required")
        return self


class TurnAction(_Body):
    kind: Literal["rerun_downstream"] = "rerun_downstream"
    artifacts: list[str] = Field(default_factory=list)


class TurnCreate(_Body):
    message: str | None = None
    mode: Literal["interactive", "auto"] | None = None
    driver: Literal["staged", "loop"] | None = None
    client_turn_id: str | None = None
    action: TurnAction | None = None
    interface_state: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _one_of(self):
        if not self.message and self.action is None:
            raise ValueError("message or action is required")
        return self


class ReplyDecision(_Body):
    prompt_id: str
    kind: Literal["answer", "option", "prerequisites", "assumption_override"] | None = None
    text: str | None = None
    index: int | None = None
    answers: dict[str, str] | None = None
    # documented aliases of the viewer's fields
    answer: str | None = None
    option_index: int | None = None
    decision: dict[str, Any] | None = None


class ROICreate(_Body):
    name: str
    slice_id: int | None = None
    type: Literal["bbox", "polygon", "circle"] = "bbox"
    params: dict[str, Any] = Field(default_factory=dict)
    modality: str | None = None


class ROIPatch(_Body):
    new_name: str


class ColorPatch(_Body):
    colors: dict[str, str]


class CelltypeRename(_Body):
    new_name: str


class SkillPatch(_Body):
    enabled: bool | None = None
    slug: str | None = None


class LLMTest(_Body):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


__all__ = ["SessionCreate", "TurnCreate", "TurnAction", "ReplyDecision", "ROICreate", "ROIPatch", "ColorPatch", "CelltypeRename", "SkillPatch", "LLMTest"]
