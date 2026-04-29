from __future__ import annotations

from src.polymarket.models import Outcome
from src.strategy.side_selection import choose_binary_side


def test_choose_yes_when_yes_has_best_edge():
    choice = choose_binary_side(
        yes_probability=0.70,
        yes_price=0.55,
        no_price=0.45,
        bankroll=1_000,
        kelly_fraction=0.25,
        max_position_usd=100,
        min_ev_threshold=0.03,
    )

    assert choice is not None
    assert choice.outcome == Outcome.YES
    assert choice.kelly.ev_per_dollar == 0.15


def test_choose_no_when_no_has_best_edge():
    choice = choose_binary_side(
        yes_probability=0.35,
        yes_price=0.55,
        no_price=0.45,
        bankroll=1_000,
        kelly_fraction=0.25,
        max_position_usd=100,
        min_ev_threshold=0.03,
    )

    assert choice is not None
    assert choice.outcome == Outcome.NO
    assert choice.kelly.ev_per_dollar == 0.20


def test_skip_when_neither_side_clears_ev_threshold():
    choice = choose_binary_side(
        yes_probability=0.51,
        yes_price=0.50,
        no_price=0.50,
        bankroll=1_000,
        kelly_fraction=0.25,
        max_position_usd=100,
        min_ev_threshold=0.03,
    )

    assert choice is None


def test_monte_carlo_gate_records_selected_side_stats():
    choice = choose_binary_side(
        yes_probability=0.70,
        yes_price=0.55,
        no_price=0.45,
        bankroll=1_000,
        kelly_fraction=0.25,
        max_position_usd=100,
        min_ev_threshold=0.03,
        use_monte_carlo=True,
        mc_samples=5_000,
        mc_seed=7,
        mc_ev_tolerance=0.03,
    )

    assert choice is not None
    assert choice.mc is not None
    assert choice.mc.samples == 5_000
    assert abs(choice.mc.mean_edge - choice.kelly.ev_per_dollar) < 0.03
