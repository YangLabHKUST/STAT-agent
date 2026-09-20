"""Chat-model factory over LangChain providers.

``make_model("poe/Claude-Sonnet-4.6")`` returns a LangChain ``BaseChatModel``. Providers:
OpenAI (native or Responses API for reasoning models), Anthropic, Google Gemini, and any
OpenAI-compatible endpoint (Poe, DeepSeek, xAI, Moonshot, Zhipu, DashScope, custom).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "poe": "https://api.poe.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "poe": ("POE_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "zhipu": ("ZHIPU_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "custom": ("STAT_API_KEY", "OPENAI_API_KEY"),
}

# Models that reject a temperature parameter (reasoning models).
NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4", "claude-opus-4-7")
RESPONSES_API_PREFIXES = ("gpt-5",)


@dataclass
class ModelSpec:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = 0.2
    max_tokens: int | None = None
    timeout: float = 600.0
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def supports_temperature(self) -> bool:
        return not self.model.lower().startswith(NO_TEMPERATURE_PREFIXES)


def infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("grok"):
        return "xai"
    if m.startswith(("moonshot", "kimi")):
        return "moonshot"
    if m.startswith("glm"):
        return "zhipu"
    if m.startswith("qwen"):
        return "dashscope"
    return "openai"


def parse_model_id(model_id: str, provider: str | None = None) -> tuple[str, str]:
    """``"poe/Claude-Sonnet-4.6"`` -> ``("poe", "Claude-Sonnet-4.6")``; bare ids infer the provider."""
    if "/" in model_id and model_id.split("/", 1)[0].lower() in ENV_KEYS | {"gemini": ()}:
        prefix, name = model_id.split("/", 1)
        prefix = prefix.lower()
        return ("google" if prefix == "gemini" else prefix), name
    return (provider or infer_provider(model_id)).lower(), model_id


def resolve_api_key(provider: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in ENV_KEYS.get(provider, ()):
        if os.environ.get(name):
            return os.environ[name]
    return None


def make_spec(
    model_id: str,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = 0.2,
    max_tokens: int | None = None,
    **extra: Any,
) -> ModelSpec:
    prov, name = parse_model_id(model_id, provider)
    if prov == "custom" and not base_url:
        raise ValueError("provider 'custom' needs a base_url")
    return ModelSpec(
        provider=prov,
        model=name,
        api_key=resolve_api_key(prov, api_key),
        base_url=base_url or OPENAI_COMPATIBLE_BASE_URLS.get(prov),
        temperature=temperature,
        max_tokens=max_tokens,
        extra=dict(extra),
    )


def make_model(spec: ModelSpec | str, **overrides: Any):
    """Instantiate the LangChain chat model for ``spec`` (a ``ModelSpec`` or a model id)."""
    if isinstance(spec, str):
        spec = make_spec(spec, **overrides)
    prov = spec.provider
    temperature = spec.temperature if spec.supports_temperature else None

    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = dict(model=spec.model, api_key=spec.api_key, max_retries=spec.max_retries, timeout=spec.timeout)
        kwargs["max_tokens"] = spec.max_tokens or 8192
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs.update(spec.extra)
        return ChatAnthropic(**kwargs)

    if prov == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = dict(model=spec.model, google_api_key=spec.api_key, max_retries=spec.max_retries, timeout=spec.timeout)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if spec.max_tokens:
            kwargs["max_output_tokens"] = spec.max_tokens
        kwargs.update(spec.extra)
        return ChatGoogleGenerativeAI(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = dict(model=spec.model, api_key=spec.api_key or "missing", max_retries=spec.max_retries, timeout=spec.timeout)
    if spec.base_url and prov != "openai":
        kwargs["base_url"] = spec.base_url
    elif spec.base_url and spec.base_url != OPENAI_COMPATIBLE_BASE_URLS["openai"]:
        kwargs["base_url"] = spec.base_url
    if temperature is not None:
        kwargs["temperature"] = temperature
    if spec.max_tokens:
        kwargs["max_tokens"] = spec.max_tokens
    if prov == "openai" and spec.model.lower().startswith(RESPONSES_API_PREFIXES):
        kwargs["use_responses_api"] = True
        kwargs.setdefault("reasoning", {"effort": spec.extra.pop("reasoning_effort", "medium")})
    if prov == "poe":
        # Poe's bots run a server-side web search by default and append a "Learn more" link list to the reply;
        # STAT wants the bare model (spec.extra can still override it)
        kwargs["extra_body"] = {"web_search": False}
        # Poe's Claude bots reason before they answer, on some prompts for ten minutes or more; a reply that sends
        # nothing for about ten minutes is cut by the server ("Connection error"). Streamed, the reasoning keeps
        # the connection alive; the reply and its usage are assembled as before.
        kwargs["streaming"] = True
        kwargs["stream_usage"] = True
        # Poe can also pause for minutes between streamed chunks; langchain-openai's default 120 s gap limit would
        # end such a call, which unstreamed waited out, so the gap limit is the request timeout
        kwargs["stream_chunk_timeout"] = spec.timeout
    kwargs.update(spec.extra)
    return ChatOpenAI(**kwargs)


__all__ = [
    "ModelSpec",
    "OPENAI_COMPATIBLE_BASE_URLS",
    "ENV_KEYS",
    "infer_provider",
    "parse_model_id",
    "resolve_api_key",
    "make_spec",
    "make_model",
]
