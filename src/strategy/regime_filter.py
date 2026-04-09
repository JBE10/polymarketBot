"""
Market regime detection and circuit breakers for spread-capture trading.

Implements lightweight technical indicators to detect when a market is
unsuitable for passive market-making (trending, volatile, or illiquid).
All math is done with pure Python — no pandas/numpy/ta-lib required.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeVerdict:
    safe: bool
    reason: str

    def __bool__(self) -> bool:
        return self.safe


_SAFE = RegimeVerdict(safe=True, reason="")


class RegimeFilter:
    """
    Evaluates whether a market is in a safe regime for spread capture.

    Checks (any failure blocks trading):
      1. Bollinger Bands — price outside 2-sigma bands → breakout
      2. ADX — strong trend (>40) → directional momentum
      3. Volume spike — current vol > 3× average → regime change
      4. Consecutive losses — too many losses in a row → halt
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_mult: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 40.0,
        volume_spike_mult: float = 3.0,
        max_consecutive_losses: int = 3,
    ) -> None:
        self._bb_period = bb_period
        self._bb_std = bb_std_mult
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._vol_spike = volume_spike_mult
        self._max_losses = max_consecutive_losses

    def is_safe(
        self,
        prices: list[float],
        volume_24h: float = 0.0,
        current_volume: float = 0.0,
        consecutive_losses: int = 0,
    ) -> RegimeVerdict:
        # --- Consecutive loss circuit breaker (cheapest check) ----------------
        if consecutive_losses >= self._max_losses:
            return RegimeVerdict(
                safe=False,
                reason=f"circuit-breaker: {consecutive_losses} consecutive losses "
                       f"(max={self._max_losses})",
            )

        if len(prices) < self._bb_period:
            return _SAFE

        # --- Bollinger Bands --------------------------------------------------
        bb = self._bollinger(prices)
        if bb is not None:
            last = prices[-1]
            if last > bb.upper or last < bb.lower:
                side = "above upper" if last > bb.upper else "below lower"
                return RegimeVerdict(
                    safe=False,
                    reason=f"bollinger: price {last:.4f} is {side} band "
                           f"[{bb.lower:.4f}, {bb.upper:.4f}]",
                )

        # --- ADX (trend strength) --------------------------------------------
        if len(prices) >= self._adx_period + 1:
            adx = self._adx(prices)
            if adx is not None and adx > self._adx_threshold:
                return RegimeVerdict(
                    safe=False,
                    reason=f"adx={adx:.1f} > {self._adx_threshold} (strong trend)",
                )

        # --- Volume spike -----------------------------------------------------
        if volume_24h > 0 and current_volume > 0:
            ratio = current_volume / volume_24h
            if ratio > self._vol_spike:
                return RegimeVerdict(
                    safe=False,
                    reason=f"volume spike: {ratio:.1f}× average "
                           f"(threshold={self._vol_spike}×)",
                )

        return _SAFE

    # ── Bollinger Bands ─────────────────────────────────────────────────────

    @dataclass(frozen=True)
    class _BB:
        sma: float
        upper: float
        lower: float

    def _bollinger(self, prices: list[float]) -> _BB | None:
        n = self._bb_period
        if len(prices) < n:
            return None
        window = prices[-n:]
        sma = sum(window) / n
        variance = sum((x - sma) ** 2 for x in window) / n
        std = math.sqrt(variance)
        return self._BB(
            sma=sma,
            upper=sma + self._bb_std * std,
            lower=sma - self._bb_std * std,
        )

    # ── ADX (Average Directional Index) ─────────────────────────────────────

    def _adx(self, prices: list[float]) -> float | None:
        """
        Simplified ADX using price changes as proxy for directional movement.

        True ADX uses high/low/close candles; here we approximate with
        sequential price differences since Polymarket only gives trade prices.
        """
        n = self._adx_period
        if len(prices) < n + 1:
            return None

        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent = changes[-n:]

        ups = [max(c, 0) for c in recent]
        downs = [max(-c, 0) for c in recent]

        avg_up = sum(ups) / n
        avg_down = sum(downs) / n
        total = avg_up + avg_down

        if total < 1e-12:
            return 0.0

        dx = abs(avg_up - avg_down) / total * 100
        return dx
