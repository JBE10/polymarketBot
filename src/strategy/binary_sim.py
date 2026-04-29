"""
Small, deterministic simulation helpers for binary prediction markets.

The strategy uses the analytical price edge (estimated probability minus entry
price) as the source of truth. Monte Carlo is used as a sanity check and to
make the short-term decision path auditable.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MonteCarloResult:
    samples: int
    mean_edge: float
    p05_edge: float
    p95_edge: float
    win_rate: float


def analytical_edge(prob: float, entry_price: float) -> float:
    """Expected per-share edge for a binary share bought at entry_price."""
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if not (0.0 < entry_price < 1.0):
        raise ValueError(f"entry_price must be in (0, 1), got {entry_price}")
    return prob - entry_price


def monte_carlo_edge(
    prob: float,
    entry_price: float,
    samples: int = 2_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Simulate per-share P&L for buying one binary share.

    If the selected side resolves true, P&L is 1 - entry_price; otherwise it is
    -entry_price. The sample mean converges to prob - entry_price.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    analytical_edge(prob, entry_price)

    rng = random.Random(seed)
    outcomes: list[float] = []
    wins = 0
    win_pnl = 1.0 - entry_price
    lose_pnl = -entry_price

    for _ in range(samples):
        if rng.random() < prob:
            outcomes.append(win_pnl)
            wins += 1
        else:
            outcomes.append(lose_pnl)

    outcomes.sort()
    p05_idx = min(samples - 1, max(0, int(samples * 0.05)))
    p95_idx = min(samples - 1, max(0, int(samples * 0.95)))

    return MonteCarloResult(
        samples=samples,
        mean_edge=round(sum(outcomes) / samples, 6),
        p05_edge=round(outcomes[p05_idx], 6),
        p95_edge=round(outcomes[p95_idx], 6),
        win_rate=round(wins / samples, 6),
    )
