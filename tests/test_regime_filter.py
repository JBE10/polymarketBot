"""
A.2 — Regime Filter unit tests.

Tests RegimeFilter.is_safe() across all detection modes:
  - Bollinger breakout (price above/below 2σ bands)
  - ADX trend detection (high directional movement)
  - Volume spike blocking
  - Consecutive loss circuit breaker
  - Safe pass-through with insufficient data
"""
from __future__ import annotations

import pytest

from src.strategy.regime_filter import RegimeFilter, RegimeVerdict


@pytest.fixture
def regime() -> RegimeFilter:
    """Default regime filter with standard params."""
    return RegimeFilter(
        bb_period=20,
        bb_std_mult=2.0,
        adx_period=14,
        adx_threshold=40.0,
        volume_spike_mult=3.0,
        max_consecutive_losses=3,
    )


# ── Insufficient data → safe ─────────────────────────────────────────────────


class TestInsufficientData:
    """When there isn't enough price data, the filter should default to SAFE."""

    def test_empty_prices(self, regime: RegimeFilter):
        verdict = regime.is_safe(prices=[], consecutive_losses=0)
        assert verdict.safe is True

    def test_fewer_than_bb_period(self, regime: RegimeFilter):
        """Less than bb_period prices → skip Bollinger/ADX, return safe."""
        prices = [0.50 + i * 0.001 for i in range(10)]  # only 10 prices
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is True

    def test_exactly_bb_period_minus_one(self, regime: RegimeFilter):
        prices = [0.50] * 19  # bb_period=20, so 19 is insufficient
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is True


# ── Consecutive loss circuit breaker ──────────────────────────────────────────


class TestCircuitBreaker:
    """Circuit breaker fires when consecutive_losses >= max."""

    def test_at_max_losses(self, regime: RegimeFilter):
        verdict = regime.is_safe(prices=[], consecutive_losses=3)
        assert verdict.safe is False
        assert "circuit-breaker" in verdict.reason

    def test_above_max_losses(self, regime: RegimeFilter):
        verdict = regime.is_safe(prices=[], consecutive_losses=5)
        assert verdict.safe is False

    def test_below_max_losses(self, regime: RegimeFilter):
        verdict = regime.is_safe(prices=[], consecutive_losses=2)
        assert verdict.safe is True

    def test_zero_losses(self, regime: RegimeFilter):
        verdict = regime.is_safe(prices=[], consecutive_losses=0)
        assert verdict.safe is True

    def test_custom_max_losses(self):
        """Custom max_consecutive_losses parameter."""
        rf = RegimeFilter(max_consecutive_losses=1)
        verdict = rf.is_safe(prices=[], consecutive_losses=1)
        assert verdict.safe is False


# ── Bollinger Band breakout detection ─────────────────────────────────────────


class TestBollingerBreakout:
    """Detect when price is outside 2σ bands."""

    def _stable_prices(self, n: int = 20, center: float = 0.50) -> list[float]:
        """Generate stable prices around a center (tight std dev)."""
        return [center + (i % 3 - 1) * 0.001 for i in range(n)]

    def test_safe_within_bands(self, regime: RegimeFilter):
        """Price near the SMA is safe."""
        prices = self._stable_prices(20, center=0.50)
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is True

    def test_breakout_above_upper_band(self, regime: RegimeFilter):
        """Price far above mean → breakout detected."""
        # 19 stable prices + 1 spike
        prices = [0.50] * 19 + [0.60]
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is False
        assert "bollinger" in verdict.reason
        assert "above upper" in verdict.reason

    def test_breakout_below_lower_band(self, regime: RegimeFilter):
        """Price far below mean → breakout detected."""
        prices = [0.50] * 19 + [0.40]
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is False
        assert "bollinger" in verdict.reason
        assert "below lower" in verdict.reason

    def test_price_exactly_at_band_is_safe(self, regime: RegimeFilter):
        """Price exactly at the band edge should still be safe (not strictly outside)."""
        # All equal prices → std=0, bands collapse to SMA
        prices = [0.50] * 20
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is True


# ── ADX trend detection ──────────────────────────────────────────────────────


class TestADXDetection:
    """Detect strong directional trends (ADX > threshold)."""

    def test_strong_uptrend_blocked(self, regime: RegimeFilter):
        """Monotonically rising prices → high ADX → blocked."""
        # Need at least adx_period + 1 = 15 prices, use 25 for Bollinger too
        prices = [0.40 + i * 0.02 for i in range(25)]  # 0.40 → 0.88
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is False
        # Could be ADX or Bollinger — either is correct for a strong trend

    def test_strong_downtrend_blocked(self, regime: RegimeFilter):
        """Monotonically falling prices → high ADX → blocked."""
        prices = [0.80 - i * 0.02 for i in range(25)]  # 0.80 → 0.32
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is False

    def test_mean_reverting_safe(self, regime: RegimeFilter):
        """Oscillating prices → low ADX → safe."""
        # Alternating up/down around 0.50
        prices = [0.50 + ((-1) ** i) * 0.005 for i in range(25)]
        verdict = regime.is_safe(prices=prices)
        assert verdict.safe is True


# ── Volume spike detection ────────────────────────────────────────────────────


class TestVolumeSpike:
    """Block trading when current volume >> 24h average."""

    def test_spike_blocked(self, regime: RegimeFilter):
        """Volume 4× the 24h average → blocked."""
        # Need >= bb_period (20) stable prices so BB check passes first
        verdict = regime.is_safe(
            prices=[0.50] * 20,
            volume_24h=10_000,
            current_volume=40_000,  # 4× > 3× threshold
        )
        assert verdict.safe is False
        assert "volume spike" in verdict.reason

    def test_normal_volume_safe(self, regime: RegimeFilter):
        """Volume 1.5× the average → safe."""
        verdict = regime.is_safe(
            prices=[0.50] * 20,
            volume_24h=10_000,
            current_volume=15_000,  # 1.5× < 3× threshold
        )
        assert verdict.safe is True

    def test_zero_volumes_safe(self, regime: RegimeFilter):
        """Zero volumes should not trigger the spike check."""
        verdict = regime.is_safe(prices=[], volume_24h=0, current_volume=0)
        assert verdict.safe is True

    def test_custom_spike_mult(self):
        """Custom volume_spike_mult parameter."""
        rf = RegimeFilter(volume_spike_mult=2.0)
        verdict = rf.is_safe(
            prices=[0.50] * 20,  # Need >= bb_period so volume check is reached
            volume_24h=10_000,
            current_volume=25_000,  # 2.5× > 2× custom threshold
        )
        assert verdict.safe is False


# ── RegimeVerdict model ───────────────────────────────────────────────────────


class TestRegimeVerdict:
    """RegimeVerdict dataclass and __bool__ semantics."""

    def test_safe_verdict_is_truthy(self):
        v = RegimeVerdict(safe=True, reason="")
        assert bool(v) is True

    def test_unsafe_verdict_is_falsy(self):
        v = RegimeVerdict(safe=False, reason="test block")
        assert bool(v) is False
        assert v.reason == "test block"

    def test_frozen_dataclass(self):
        v = RegimeVerdict(safe=True, reason="")
        with pytest.raises(AttributeError):
            v.safe = False  # type: ignore[misc]
