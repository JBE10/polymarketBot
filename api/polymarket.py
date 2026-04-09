"""
Async Polymarket API client.
Covers: Gamma REST API (markets/events), CLOB API (order books/trades), Data API (positions).
All endpoints used here are public and require no authentication.
"""
from __future__ import annotations

from typing import Any

import httpx

from config import (
    MAX_MARKETS_FETCH,
    POLYMARKET_CLOB_API,
    POLYMARKET_DATA_API,
    POLYMARKET_GAMMA_API,
    REQUEST_TIMEOUT,
)


class PolymarketClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────
    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers={"Accept": "application/json", "User-Agent": "polymarket-intel/1.0"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, base: str, path: str, params: dict | None = None) -> Any:
        client = await self._http()
        try:
            resp = await client.get(f"{base}{path}", params=params or {})
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"HTTP {exc.response.status_code}: {base}{path}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Request failed: {exc}") from exc

    # ── Gamma API — market listings ───────────────────────────────────────────
    async def get_markets(
        self,
        limit: int = MAX_MARKETS_FETCH,
        offset: int = 0,
        sort_by: str = "volume24hr",
    ) -> list[dict[str, Any]]:
        """Fetch active, non-closed markets sorted by the given field."""
        try:
            data = await self._get(
                POLYMARKET_GAMMA_API,
                "/markets",
                {
                    "limit": limit,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "_sort": sort_by,
                    "_order": "DESC",
                },
            )
            return data if isinstance(data, list) else data.get("markets", [])
        except Exception:
            return []

    async def get_market(self, market_id: str) -> dict[str, Any] | None:
        try:
            return await self._get(POLYMARKET_GAMMA_API, f"/markets/{market_id}")
        except Exception:
            return None

    async def search_markets(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        try:
            data = await self._get(
                POLYMARKET_GAMMA_API,
                "/markets",
                {"_q": query, "limit": limit, "active": "true", "closed": "false"},
            )
            return data if isinstance(data, list) else data.get("markets", [])
        except Exception:
            return []

    # ── CLOB API — order books & trades ──────────────────────────────────────
    async def get_order_book(self, token_id: str) -> dict[str, Any] | None:
        """Fetch the live order book for a YES or NO token."""
        try:
            return await self._get(POLYMARKET_CLOB_API, "/book", {"token_id": token_id})
        except Exception:
            return None

    async def get_trades(
        self, token_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch the most recent trades for a token."""
        try:
            data = await self._get(
                POLYMARKET_CLOB_API,
                "/trades",
                {"token_id": token_id, "limit": limit},
            )
            return data if isinstance(data, list) else data.get("data", [])
        except Exception:
            return []

    async def get_prices_history(
        self, token_id: str, interval: str = "1d", fidelity: int = 60
    ) -> list[dict[str, Any]]:
        """Fetch price history for a token (interval: 1h, 1d, 1w, 1m, all)."""
        try:
            data = await self._get(
                POLYMARKET_CLOB_API,
                "/prices-history",
                {"token_id": token_id, "interval": interval, "fidelity": fidelity},
            )
            return data.get("history", [])
        except Exception:
            return []

    async def get_clob_market(self, condition_id: str) -> dict[str, Any] | None:
        """Fetch CLOB market metadata for a condition ID."""
        try:
            return await self._get(POLYMARKET_CLOB_API, f"/markets/{condition_id}")
        except Exception:
            return None

    # ── Data API — wallet positions ───────────────────────────────────────────
    async def get_positions(self, address: str) -> list[dict[str, Any]]:
        """
        Fetch open positions for a wallet address.
        Returns portfolio entries with market question, outcome, size, avg price, current value.
        No private keys are used — address is public on-chain.
        """
        try:
            data = await self._get(
                POLYMARKET_DATA_API,
                "/positions",
                {"user": address, "sizeThreshold": "0.01"},
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    async def get_activity(
        self, address: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch on-chain trading activity for a wallet address."""
        try:
            data = await self._get(
                POLYMARKET_DATA_API,
                "/activity",
                {"user": address, "limit": limit},
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    async def get_value(self, address: str) -> dict[str, Any] | None:
        """Fetch total portfolio value for a wallet address."""
        try:
            data = await self._get(
                POLYMARKET_DATA_API, "/value", {"user": address}
            )
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    # ── Convenience: multi-market trades for whale scanning ──────────────────
    async def get_recent_trades_bulk(
        self, markets: list[dict[str, Any]], per_market: int = 30
    ) -> list[dict[str, Any]]:
        """
        Fetch trades across multiple markets for whale detection.
        Attaches 'question' field to each trade for display.
        Only uses the YES token (first clobTokenId) per market.
        """
        all_trades: list[dict[str, Any]] = []
        for market in markets:
            token_ids: list[str] = market.get("clobTokenIds") or []
            if not token_ids:
                continue
            trades = await self.get_trades(token_ids[0], limit=per_market)
            for t in trades:
                t["_question"] = market.get("question", "")[:60]
                t["_market_id"] = market.get("id", "")
            all_trades.extend(trades)
        return all_trades


# Module-level singleton
poly_client = PolymarketClient()
