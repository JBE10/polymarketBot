"""
Phase 1 — Kelly Criterion mathematical core.

Validates that compute_kelly(), expected_value(), kelly_criterion(), and
min_prob_for_positive_ev() produce correct results for known inputs.
"""
from __future__ import annotations

import pytest

from src.strategy.kelly import (
    compute_kelly,
    expected_value,
    kelly_criterion,
    max_loss,
    min_prob_for_positive_ev,
    multi_market_kelly,
)

# ── Constants for the canonical test case ─────────────────────────────────────

PROB       = 0.88
PRICE      = 0.67
BANKROLL   = 10_000.0
KELLY_FRAC = 0.25
TOL        = 1e-3


class TestComputeKelly:
    """compute_kelly() — full computation returning KellyResult."""

    def test_canonical_case(self):
        """Expected: b≈0.493, full_kelly≈0.636, frac≈0.159, position≈$1590."""
        kr = compute_kelly(
            prob=PROB,
            entry_price=PRICE,
            bankroll=BANKROLL,
            kelly_fraction=KELLY_FRAC,
            max_position_usd=5_000,
        )

        # Manual math
        b = (1.0 - PRICE) / PRICE           # ≈ 0.49254
        q = 1.0 - PROB                       # 0.12
        full_kelly = (b * PROB - q) / b      # ≈ 0.63636
        frac_kelly = full_kelly * KELLY_FRAC # ≈ 0.15909

        assert abs(kr.frac_kelly - frac_kelly) < TOL
        assert abs(kr.frac_kelly - 0.1591) < TOL
        assert kr.is_positive_ev
        assert kr.ev_per_dollar > 0

        expected_pos = frac_kelly * BANKROLL  # ~$1,590.91
        assert abs(kr.position_usd - expected_pos) < 1.0

    def test_no_edge(self):
        """50% prob at 60¢ should have no edge."""
        kr = compute_kelly(prob=0.50, entry_price=0.60, bankroll=1000)
        assert not kr.is_positive_ev
        assert kr.frac_kelly == 0.0

    def test_max_position_cap(self):
        """Position size should be capped by max_position_usd."""
        kr = compute_kelly(
            prob=0.95, entry_price=0.50,
            bankroll=100_000, max_position_usd=500.0,
        )
        assert kr.position_usd <= 500.0

    def test_invalid_price_raises(self):
        """Price outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError):
            compute_kelly(prob=0.5, entry_price=0.0, bankroll=1000)
        with pytest.raises(ValueError):
            compute_kelly(prob=0.5, entry_price=1.0, bankroll=1000)

    def test_shares_calculation(self):
        """shares = position_usd / entry_price."""
        kr = compute_kelly(
            prob=0.80, entry_price=0.50,
            bankroll=10_000, kelly_fraction=0.25,
        )
        expected_shares = kr.position_usd / 0.50
        assert abs(kr.shares - expected_shares) < 0.01


class TestExpectedValue:
    """expected_value() — EV per dollar staked."""

    def test_positive_ev(self):
        ev = expected_value(PROB, PRICE)
        assert abs(ev - 0.21) < TOL

    def test_negative_ev(self):
        ev = expected_value(0.40, 0.60)
        assert ev < 0

    def test_breakeven(self):
        ev = expected_value(0.50, 0.50)
        assert abs(ev) < TOL


class TestMinProbForPositiveEV:
    """min_prob_for_positive_ev() — breakeven probability equals price."""

    def test_breakeven_equals_price(self):
        breakeven = min_prob_for_positive_ev(PRICE)
        assert abs(breakeven - PRICE) < TOL


class TestKellyCriterion:
    """kelly_criterion() — fractional Kelly as a float."""

    def test_returns_zero_for_no_edge(self):
        frac = kelly_criterion(prob=0.30, entry_price=0.50)
        assert frac == 0.0

    def test_returns_positive_for_edge(self):
        frac = kelly_criterion(prob=0.80, entry_price=0.50, kelly_fraction=0.25)
        assert frac > 0

    def test_capped_at_20_percent(self):
        """Hard cap at 20% of bankroll even with enormous edge."""
        frac = kelly_criterion(prob=0.99, entry_price=0.10, kelly_fraction=1.0)
        assert frac <= 0.20


class TestMaxLoss:
    """max_loss() — worst case = entire stake."""

    def test_max_loss_equals_position(self):
        assert max_loss(250.0) == 250.0


class TestMultiMarketKelly:
    """multi_market_kelly() — naive multi-bet sizing with normalisation."""

    def test_normalises_exceeding_bankroll(self):
        opportunities = [(0.90, 0.50), (0.85, 0.40), (0.88, 0.45)]
        sizes = multi_market_kelly(opportunities, bankroll=1000, kelly_fraction=0.25)
        assert sum(sizes) <= 1000.0 + 0.01  # allow float rounding

    def test_zero_edge_gets_zero_size(self):
        opportunities = [(0.30, 0.50)]  # no edge
        sizes = multi_market_kelly(opportunities, bankroll=1000)
        assert sizes[0] == 0.0
