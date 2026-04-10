"""
Short-term market intelligence module.

Provides real-time contextual signals specifically designed for ultra-short
prediction markets like "Will BTC go up in the next 10 minutes?":

  - Live crypto prices (via CryptoPriceFetcher / Binance)
  - Recent price momentum (short-window % change)
  - Order book imbalance (OBI) from the Binance order book
  - Market volatility estimate
  - Human-readable "market regime" classification

This data is injected into LLM prompts to enable better decisions on
short-lived pools without needing any external paid API.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Binance endpoints (no key required) ──────────────────────────────────────
_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"
_BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
_BINANCE_AGG_TRADES_URL = "https://api.binance.com/api/v3/aggTrades?symbol={symbol}&limit=50"

_CACHE_TTL_S = 15  # seconds — refresh every 15 s for short-term accuracy


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CryptoSignal:
    symbol: str
    price: float
    change_1m_pct: float = 0.0    # momentum last 1 minute
    change_5m_pct: float = 0.0    # momentum last 5 minutes
    change_15m_pct: float = 0.0   # momentum last 15 minutes
    change_1h_pct: float = 0.0
    change_24h_pct: float = 0.0
    obi: float = 0.0              # order book imbalance [-1, +1]  positive = buy pressure
    volatility_pct: float = 0.0   # recent volatility proxy
    volume_spike_ratio: float = 1.0  # current volume vs average
    regime: str = "neutral"       # bullish / bearish / neutral / volatile
    timestamp: float = field(default_factory=time.monotonic)
    source: str = "Binance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_1m_pct": self.change_1m_pct,
            "change_5m_pct": self.change_5m_pct,
            "change_15m_pct": self.change_15m_pct,
            "change_1h_pct": self.change_1h_pct,
            "change_24h_pct": self.change_24h_pct,
            "obi": self.obi,
            "volatility_pct": self.volatility_pct,
            "volume_spike_ratio": self.volume_spike_ratio,
            "regime": self.regime,
        }


# ── In-memory signal cache ─────────────────────────────────────────────────

_signal_cache: dict[str, tuple[CryptoSignal, float]] = {}  # symbol → (signal, mono_ts)


def _is_fresh(symbol: str) -> bool:
    if symbol not in _signal_cache:
        return False
    _, ts = _signal_cache[symbol]
    return (time.monotonic() - ts) < _CACHE_TTL_S


# ── Core fetcher ──────────────────────────────────────────────────────────────

class ShortTermMarketContext:
    """
    Asynchronous fetcher for real-time crypto signals.

    Usage::

        ctx = ShortTermMarketContext()
        context_text = await ctx.get_context_for_market(
            "Will BTC go up in the next 10 minutes?"
        )
        await ctx.close()
    """

    def __init__(self, timeout_s: int = 5) -> None:
        self._http = httpx.AsyncClient(timeout=float(timeout_s))

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_signal(self, symbol: str) -> CryptoSignal | None:
        """Return a CryptoSignal for the given symbol (e.g. 'BTC')."""
        sym = symbol.upper()
        if _is_fresh(sym):
            return _signal_cache[sym][0]

        signal = await self._build_signal(sym)
        if signal:
            _signal_cache[sym] = (signal, time.monotonic())
        return signal

    async def get_context_for_market(self, market_question: str) -> str:
        """
        Extract relevant symbols from the market question and return a
        formatted context string suitable for prompt injection.
        """
        symbols = self._extract_symbols(market_question)
        if not symbols:
            return ""

        signals = await asyncio.gather(
            *[self.get_signal(s) for s in symbols],
            return_exceptions=True,
        )

        parts: list[str] = []
        for sig in signals:
            if isinstance(sig, CryptoSignal):
                parts.append(self._format_signal(sig))

        if not parts:
            return ""

        header = "=== Short-Term Crypto Intelligence (real-time) ==="
        footer = "(Data from Binance public API, updated every 15 s)"
        return "\n".join([header] + parts + [footer])

    async def close(self) -> None:
        await self._http.aclose()

    # ── Signal building ───────────────────────────────────────────────────────

    async def _build_signal(self, symbol: str) -> CryptoSignal | None:
        pair = f"{symbol}USDT"
        try:
            # Parallel fetch: klines, depth, 24h ticker
            klines_task = self._fetch_klines(pair)
            depth_task = self._fetch_depth(pair)
            ticker_task = self._fetch_ticker(pair)

            klines, depth, ticker = await asyncio.gather(
                klines_task, depth_task, ticker_task, return_exceptions=True
            )

            price = 0.0
            change_24h = 0.0
            if not isinstance(ticker, Exception) and ticker:
                price = float(ticker.get("lastPrice", 0))
                change_24h = float(ticker.get("priceChangePercent", 0))

            if price == 0.0:
                return None

            # Momentum from klines
            c1m = c5m = c15m = c1h = 0.0
            volatility = 0.0
            vol_spike = 1.0
            if not isinstance(klines, Exception) and klines:
                c1m, c5m, c15m, c1h, volatility, vol_spike = self._analyse_klines(
                    klines, price
                )

            # OBI from order book
            obi = 0.0
            if not isinstance(depth, Exception) and depth:
                obi = self._calc_obi(depth)

            regime = self._classify_regime(c5m, c15m, obi, volatility)

            return CryptoSignal(
                symbol=symbol,
                price=price,
                change_1m_pct=c1m,
                change_5m_pct=c5m,
                change_15m_pct=c15m,
                change_1h_pct=c1h,
                change_24h_pct=change_24h,
                obi=obi,
                volatility_pct=volatility,
                volume_spike_ratio=vol_spike,
                regime=regime,
            )

        except Exception as exc:
            log.debug("ShortTermMarketContext._build_signal(%s): %s", symbol, exc)
            return None

    # ── Binance data fetchers ─────────────────────────────────────────────────

    async def _fetch_klines(self, pair: str) -> list:
        """Fetch 1-minute klines for the past hour (60 candles)."""
        params = {"symbol": pair, "interval": "1m", "limit": 65}
        resp = await self._http.get(_BINANCE_KLINES_URL, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _fetch_depth(self, pair: str) -> dict:
        """Fetch top-20 order book levels."""
        params = {"symbol": pair, "limit": 20}
        resp = await self._http.get(_BINANCE_DEPTH_URL, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _fetch_ticker(self, pair: str) -> dict:
        url = _BINANCE_TICKER_URL.format(symbol=pair)
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _analyse_klines(
        self, klines: list, current_price: float
    ) -> tuple[float, float, float, float, float, float]:
        """
        Returns (change_1m, change_5m, change_15m, change_1h, volatility, vol_spike_ratio).
        klines is a list of [open_time, open, high, low, close, volume, …]
        """
        if not klines:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        def pct_chg(from_price: float) -> float:
            if from_price == 0:
                return 0.0
            return (current_price - from_price) / from_price * 100

        c1m = pct_chg(closes[-2]) if len(closes) >= 2 else 0.0
        c5m = pct_chg(closes[-6]) if len(closes) >= 6 else 0.0
        c15m = pct_chg(closes[-16]) if len(closes) >= 16 else 0.0
        c1h = pct_chg(closes[-61]) if len(closes) >= 61 else 0.0

        # Volatility: std of last 15 minute % changes
        if len(closes) >= 16:
            pcts = [
                (closes[i] - closes[i - 1]) / closes[i - 1] * 100
                for i in range(len(closes) - 15, len(closes))
                if closes[i - 1] != 0
            ]
            if pcts:
                mean = sum(pcts) / len(pcts)
                variance = sum((p - mean) ** 2 for p in pcts) / len(pcts)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
        else:
            volatility = 0.0

        # Volume spike: last candle vs 10-candle avg
        if len(volumes) >= 11:
            avg_vol = sum(volumes[-11:-1]) / 10
            vol_spike = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_spike = 1.0

        return c1m, c5m, c15m, c1h, volatility, vol_spike

    def _calc_obi(self, depth: dict) -> float:
        """
        Order Book Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Result in [-1, +1].  Positive = buy pressure, Negative = sell pressure.
        """
        try:
            bid_vol = sum(float(b[1]) for b in depth.get("bids", []))
            ask_vol = sum(float(a[1]) for a in depth.get("asks", []))
            total = bid_vol + ask_vol
            if total == 0:
                return 0.0
            return (bid_vol - ask_vol) / total
        except Exception:
            return 0.0

    def _classify_regime(
        self, c5m: float, c15m: float, obi: float, volatility: float
    ) -> str:
        """Simple rule-based regime classification."""
        if volatility > 0.5:
            return "volatile"
        if c5m > 0.3 and obi > 0.1:
            return "bullish"
        if c5m < -0.3 and obi < -0.1:
            return "bearish"
        if c15m > 0.5:
            return "bullish"
        if c15m < -0.5:
            return "bearish"
        return "neutral"

    # ── Formatting ─────────────────────────────────────────────────────────────

    def _format_signal(self, s: CryptoSignal) -> str:
        trend_arrow = "▲" if s.change_5m_pct >= 0 else "▼"
        obi_bar = self._obi_bar(s.obi)
        vol_note = ""
        if s.volume_spike_ratio > 1.5:
            vol_note = f"  ⚡ Vol spike {s.volume_spike_ratio:.1f}x"

        lines = [
            f"\n  {s.symbol} @ ${s.price:,.2f}  [{s.regime.upper()}]",
            f"    Momentum:  1m={s.change_1m_pct:+.2f}%  5m={s.change_5m_pct:+.2f}%  "
            f"15m={s.change_15m_pct:+.2f}%  1h={s.change_1h_pct:+.2f}%  "
            f"24h={s.change_24h_pct:+.2f}%  {trend_arrow}",
            f"    OBI:       {obi_bar}  ({s.obi:+.3f})  "
            f"Volatility(15m): {s.volatility_pct:.3f}%{vol_note}",
        ]

        # Interpretation hint for the LLM
        hint = self._interpretation_hint(s)
        if hint:
            lines.append(f"    ⟹ {hint}")

        return "\n".join(lines)

    def _obi_bar(self, obi: float, width: int = 10) -> str:
        """Render OBI as a simple ASCII bar: ■■■■■□□□□□"""
        filled = round((obi + 1) / 2 * width)
        filled = max(0, min(width, filled))
        return "■" * filled + "□" * (width - filled)

    def _interpretation_hint(self, s: CryptoSignal) -> str:
        if s.regime == "bullish" and s.change_5m_pct > 0.5:
            return "Strong upward momentum — consider YES on 'price goes up' pools"
        if s.regime == "bearish" and s.change_5m_pct < -0.5:
            return "Strong downward momentum — consider YES on 'price goes down' pools"
        if s.regime == "volatile":
            return "High volatility — directional prediction is uncertain; price pools are risky"
        if abs(s.obi) > 0.3:
            direction = "buyers" if s.obi > 0 else "sellers"
            return f"Order book dominated by {direction} (OBI={s.obi:+.2f})"
        return ""

    # ── Symbol extraction ─────────────────────────────────────────────────────

    _KNOWN_SYMBOLS = {
        "BTC", "BITCOIN",
        "ETH", "ETHEREUM",
        "SOL", "SOLANA",
        "BNB",
        "MATIC", "POLYGON",
        "AVAX", "AVALANCHE",
        "DOGE", "DOGECOIN",
        "XRP", "RIPPLE",
        "ADA", "CARDANO",
        "DOT", "POLKADOT",
        "LINK", "CHAINLINK",
        "LTC", "LITECOIN",
    }

    _SYMBOL_ALIASES = {
        "BITCOIN": "BTC",
        "ETHEREUM": "ETH",
        "SOLANA": "SOL",
        "POLYGON": "MATIC",
        "AVALANCHE": "AVAX",
        "DOGECOIN": "DOGE",
        "RIPPLE": "XRP",
        "CARDANO": "ADA",
        "POLKADOT": "DOT",
        "CHAINLINK": "LINK",
        "LITECOIN": "LTC",
    }

    def _extract_symbols(self, text: str) -> list[str]:
        """
        Heuristically extract crypto symbols from a market question string.
        Returns up to 3 most relevant symbols.
        """
        upper = text.upper()
        found: list[str] = []
        for token in self._KNOWN_SYMBOLS:
            if token in upper:
                canonical = self._SYMBOL_ALIASES.get(token, token)
                if canonical not in found:
                    found.append(canonical)
        # Limit to 3 to keep prompts concise
        return found[:3]
