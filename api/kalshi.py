"""
Async Kalshi API client (public endpoints — no authentication required).
Used primarily for cross-platform arbitrage comparison.
"""
from __future__ import annotations

from typing import Any

import httpx

from config import KALSHI_API_BASE, REQUEST_TIMEOUT


class KalshiClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

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

    async def _get(self, path: str, params: dict | None = None) -> Any:
        client = await self._http()
        try:
            resp = await client.get(f"{KALSHI_API_BASE}{path}", params=params or {})
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Kalshi HTTP {exc.response.status_code}: {path}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Kalshi request failed: {exc}") from exc

    # ── Market listings ───────────────────────────────────────────────────────
    async def get_markets(
        self, limit: int = 200, status: str = "open"
    ) -> list[dict[str, Any]]:
        """Fetch open Kalshi markets."""
        try:
            data = await self._get("/markets", {"limit": limit, "status": status})
            return data.get("markets", [])
        except Exception:
            return []

    async def get_market(self, ticker: str) -> dict[str, Any] | None:
        """Fetch a single Kalshi market by ticker."""
        try:
            data = await self._get(f"/markets/{ticker}")
            return data.get("market")
        except Exception:
            return None

    async def get_orderbook(
        self, ticker: str, depth: int = 10
    ) -> dict[str, Any] | None:
        """Fetch order book for a Kalshi market (if publicly accessible)."""
        try:
            data = await self._get(
                f"/markets/{ticker}/orderbook", {"depth": depth}
            )
            return data.get("orderbook")
        except Exception:
            return None

    async def get_series(self, series_ticker: str) -> dict[str, Any] | None:
        """Fetch a Kalshi series (group of related markets)."""
        try:
            data = await self._get(f"/series/{series_ticker}")
            return data.get("series")
        except Exception:
            return None

    # ── Helper: YES price extraction ──────────────────────────────────────────
    @staticmethod
    def extract_yes_price(market: dict[str, Any]) -> float | None:
        """
        Kalshi prices are integers in cents (0–99).
        Returns a float in [0, 1] suitable for comparison with Polymarket.
        """
        try:
            for field in ("yes_bid", "yes_ask", "last_price", "yes_sub_title"):
                val = market.get(field)
                if val is not None:
                    return float(val) / 100.0
        except (ValueError, TypeError):
            pass
        return None


# Module-level singleton
kalshi_client = KalshiClient()
