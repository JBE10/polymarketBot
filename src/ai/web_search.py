"""
Real-time crypto price context for market evaluation.

Fetches live prices and 24h momentum from public APIs (Binance + CoinGecko)
without requiring any API key.  The resulting formatted text is injected into
the LLM prompt so the model has up-to-date price context for short-term
crypto prediction markets (e.g. "Will BTC go up in the next 10 min?").

Drop-in replacement for the old Tavily-based WebSearcher — the public
interface (is_available / search / close) is preserved so callers need
no changes.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Public API endpoints (no key required) ────────────────────────────────────
_BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
_COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_last_updated_at=true"
)

# Map symbol → CoinGecko id (extend as needed)
_SYMBOL_TO_CG_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "MATIC": "matic-network",
    "USDC": "usd-coin",
    "BNB": "binancecoin",
    "AVAX": "avalanche-2",
    "DOGE": "dogecoin",
}

# Cache to avoid hammering the API every single call
_CACHE_TTL_S = 20  # seconds


class _PriceCache:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._ts: float = 0.0

    def is_fresh(self) -> bool:
        return (time.monotonic() - self._ts) < _CACHE_TTL_S

    def set(self, data: dict[str, Any]) -> None:
        self._data = data
        self._ts = time.monotonic()

    def get(self) -> dict[str, Any]:
        return self._data


_cache = _PriceCache()


class CryptoPriceFetcher:
    """
    Fetches live BTC/ETH/SOL/… prices and formats them as LLM-ready context.

    Tries Binance first (faster, more granular).  Falls back to CoinGecko.
    No API key required for either endpoint.
    """

    def __init__(
        self,
        symbols: str = "BTC,ETH,SOL,MATIC",
        timeout_s: int = 5,
        enabled: bool = True,
    ) -> None:
        self._symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        self._timeout = timeout_s
        self._enabled = enabled
        self._http = httpx.AsyncClient(timeout=float(timeout_s))

    # ── Public interface (mirrors old WebSearcher) ────────────────────────────

    def is_available(self) -> bool:
        return self._enabled

    async def search(self, query: str, max_results: int | None = None) -> str:
        """
        Legacy interface.  `query` is ignored; returns live price context.
        `max_results` is unused (kept for API compatibility).
        """
        return await self.get_context()

    async def get_context(self) -> str:
        """Return a formatted price context string ready for prompt injection."""
        if not self._enabled:
            return ""

        if _cache.is_fresh():
            return self._format(_cache.get())

        data = await self._fetch()
        if data:
            _cache.set(data)
            return self._format(data)
        return ""

    async def close(self) -> None:
        await self._http.aclose()

    # ── Fetching ──────────────────────────────────────────────────────────────

    async def _fetch(self) -> dict[str, Any]:
        """Try Binance; fall back to CoinGecko."""
        data = await self._fetch_binance()
        if not data:
            data = await self._fetch_coingecko()
        return data

    async def _fetch_binance(self) -> dict[str, Any]:
        """
        Binance /ticker/24hr gives price, 24h change%, high/low and volume.
        Returns a dict keyed by symbol, or {} on failure.
        """
        try:
            resp = await self._http.get(_BINANCE_TICKER_URL)
            resp.raise_for_status()
            tickers: list[dict] = resp.json()

            # Index by base asset (strip USDT suffix)
            result: dict[str, Any] = {}
            for t in tickers:
                symbol = t.get("symbol", "")
                if not symbol.endswith("USDT"):
                    continue
                base = symbol[:-4]  # e.g. "BTC"
                if base not in self._symbols:
                    continue
                result[base] = {
                    "price": float(t.get("lastPrice", 0)),
                    "change_24h_pct": float(t.get("priceChangePercent", 0)),
                    "high_24h": float(t.get("highPrice", 0)),
                    "low_24h": float(t.get("lowPrice", 0)),
                    "volume_24h_usd": float(t.get("quoteVolume", 0)),
                    "source": "Binance",
                }
            return result

        except Exception as exc:
            log.debug("Binance ticker fetch failed: %s", exc)
            return {}

    async def _fetch_coingecko(self) -> dict[str, Any]:
        """
        CoinGecko /simple/price — free, no key, rate-limited to ~30 req/min.
        Returns a dict keyed by symbol, or {} on failure.
        """
        cg_ids = [
            _SYMBOL_TO_CG_ID[s]
            for s in self._symbols
            if s in _SYMBOL_TO_CG_ID
        ]
        if not cg_ids:
            return {}

        try:
            url = _COINGECKO_URL.format(ids=",".join(cg_ids))
            resp = await self._http.get(url)
            resp.raise_for_status()
            raw: dict = resp.json()

            # Reverse map: cg_id → symbol
            cg_to_sym = {v: k for k, v in _SYMBOL_TO_CG_ID.items()}

            result: dict[str, Any] = {}
            for cg_id, vals in raw.items():
                sym = cg_to_sym.get(cg_id)
                if sym and sym in self._symbols:
                    result[sym] = {
                        "price": float(vals.get("usd", 0)),
                        "change_24h_pct": float(vals.get("usd_24h_change", 0)),
                        "high_24h": None,
                        "low_24h": None,
                        "volume_24h_usd": None,
                        "source": "CoinGecko",
                    }
            return result

        except Exception as exc:
            log.debug("CoinGecko fetch failed: %s", exc)
            return {}

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format(self, data: dict[str, Any]) -> str:
        if not data:
            return ""

        lines = ["=== Live Crypto Prices (real-time) ==="]
        for sym in self._symbols:
            if sym not in data:
                continue
            d = data[sym]
            price = d["price"]
            chg = d["change_24h_pct"]
            trend = "▲" if chg >= 0 else "▼"
            h24 = f"  H:{d['high_24h']:,.2f}" if d["high_24h"] else ""
            l24 = f"  L:{d['low_24h']:,.2f}" if d["low_24h"] else ""
            vol = f"  Vol(24h):${d['volume_24h_usd']:,.0f}" if d["volume_24h_usd"] else ""
            src = d.get("source", "")
            lines.append(
                f"  {sym}: ${price:,.2f}  {trend}{abs(chg):.2f}%{h24}{l24}{vol}  [{src}]"
            )

        lines.append("(Prices updated within the last 20 seconds)")
        return "\n".join(lines)


# ── Backward-compat alias ─────────────────────────────────────────────────────
# Old code may import `WebSearcher` directly.
WebSearcher = CryptoPriceFetcher
