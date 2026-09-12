"""Optional web/doc lookup tool for the method-researcher subagent (``research`` extra).

Returns ``None`` when no provider is configured so the caller can run the subagent without
tools; nothing here is imported by the staged pipeline.
"""

from __future__ import annotations

import os
from typing import Any


def web_search_tool() -> Any | None:
    """A LangChain tool for web search when ``TAVILY_API_KEY`` and ``tavily-python`` are present."""
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    try:
        from langchain_core.tools import tool
        from tavily import TavilyClient
    except ImportError:
        return None
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web for a method's documentation, requirements and typical outputs."""
        res = client.search(query=query, max_results=max_results, include_answer=True)
        parts = []
        if res.get("answer"):
            parts.append(f"Answer: {res['answer']}")
        for r in res.get("results", []):
            parts.append(f"- {r.get('title')} ({r.get('url')}): {(r.get('content') or '')[:400]}")
        return "\n".join(parts) or "no results"

    return web_search


def fetch_url_tool() -> Any | None:
    """A minimal page fetcher (httpx) for README/doc pages named by the user."""
    try:
        import httpx
        from langchain_core.tools import tool
    except ImportError:
        return None

    @tool
    def fetch_url(url: str, max_chars: int = 8000) -> str:
        """Fetch a documentation page or README (text only, truncated)."""
        try:
            r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "stat-agent/2"})
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return f"fetch failed: {exc}"
        text = r.text
        if "<html" in text[:2000].lower():
            import re

            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
        return text[:max_chars]

    return fetch_url


__all__ = ["web_search_tool", "fetch_url_tool"]
