from __future__ import annotations

import pytest

from src.strategy.binary_sim import analytical_edge, monte_carlo_edge


def test_analytical_edge_is_probability_minus_price():
    assert analytical_edge(0.62, 0.55) == pytest.approx(0.07)


def test_monte_carlo_mean_converges_to_analytical_edge():
    result = monte_carlo_edge(prob=0.62, entry_price=0.55, samples=20_000, seed=7)

    assert result.mean_edge == pytest.approx(analytical_edge(0.62, 0.55), abs=0.01)
    assert result.win_rate == pytest.approx(0.62, abs=0.01)
