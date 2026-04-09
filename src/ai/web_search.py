"""
Real-time web search for market evaluation context.

Searches the internet via Tavily API (free tier: 1,000 searches/month)
and returns formatted snippets that get injected into the LLM prompt
so the model has up-to-date information rather than stale training data.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


class WebSearcher:
    def __init__(
        self,
        api_key: str = "",
        max_results: int = 5,
        enabled: bool = True,
    ) -> None:
        self._api_key = api_key.strip()
        self._max_results = max_results
        self._enabled = enabled and bool(self._api_key)
        self._http = httpx.AsyncClient(timeout=30.0)

    @property
    def is_available(self) -> bool:
        return self._enabled

    async def search(self, query: str, max_results: int | None = None) -> str:
        """
        Search the web and return formatted text ready for prompt injection.

        Returns empty string if disabled, no API key, or on error.
        """
        if not self._enabled or not query.strip():
            return ""

        n = max_results or self._max_results

        try:
            payload = {
                "api_key": self._api_key,
                "query": query,
                "max_results": n,
                "search_depth": "basic",
                "include_answer": True,
            }

            resp = await self._http.post(_TAVILY_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            return self._format_results(data, query)

        except httpx.HTTPStatusError as exc:
            log.warning(
                "Tavily HTTP %d for query '%s': %s",
                exc.response.status_code, query[:60],
                exc.response.text[:200],
            )
            return ""
        except Exception as exc:
            log.warning("Web search failed for '%s': %s", query[:60], exc)
            return ""

    def _format_results(self, data: dict, query: str) -> str:
        snippets: list[str] = []

        answer = data.get("answer")
        if answer:
            snippets.append(f"**AI Summary:** {answer}")

        results = data.get("results", [])
        for i, r in enumerate(results, start=1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            published = r.get("published_date", "")

            date_str = f" ({published})" if published else ""
            snippets.append(
                f"[{i}] {title}{date_str}\n{content}\nSource: {url}"
            )

        if not snippets:
            return ""

        return "\n\n---\n\n".join(snippets)

    async def close(self) -> None:
        await self._http.aclose()
