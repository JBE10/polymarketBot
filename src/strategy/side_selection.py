"""
YES/NO selection for short-duration binary markets.

The LLM estimates P(YES). We convert that into two possible trades:
buy YES with probability p, or buy NO with probability 1-p. The chosen side is
the one with the best positive EV after Kelly sizing and optional MC validation.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.polymarket.models import Outcome
from src.strategy.binary_sim import MonteCarloResult, monte_carlo_edge
from src.strategy.kelly import KellyResult, compute_kelly


@dataclass(frozen=True)
class SideChoice:
    outcome: Outcome
    price: float
    prob: float
    kelly: KellyResult
    other_ev_per_dollar: float
    mc: MonteCarloResult | None = None


def choose_binary_side(
    *,
    yes_probability: float,
    yes_price: float,
    no_price: float,
    bankroll: float,
    kelly_fraction: float,
    max_position_usd: float,
    min_ev_threshold: float,
    use_monte_carlo: bool = False,
    mc_samples: int = 2_000,
    mc_seed: int | None = None,
    mc_ev_tolerance: float = 0.01,
) -> SideChoice | None:
    """Return the best YES/NO side, or None when neither clears risk gates."""
    yes = compute_kelly(
        prob=yes_probability,
        entry_price=yes_price,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_position_usd=max_position_usd,
    )
    no = compute_kelly(
        prob=1.0 - yes_probability,
        entry_price=no_price,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_position_usd=max_position_usd,
    )

    eligible: list[tuple[Outcome, float, KellyResult, float]] = []
    if _eligible(yes, min_ev_threshold):
        eligible.append((Outcome.YES, yes_price, yes, no.ev_per_dollar))
    if _eligible(no, min_ev_threshold):
        eligible.append((Outcome.NO, no_price, no, yes.ev_per_dollar))
    if not eligible:
        return None

    outcome, price, kelly, other_ev = max(
        eligible,
        key=lambda item: (
            item[2].ev_per_dollar,
            item[2].expected_profit,
            item[2].position_usd,
        ),
    )

    mc_result = None
    if use_monte_carlo:
        mc_result = monte_carlo_edge(
            prob=kelly.prob,
            entry_price=price,
            samples=mc_samples,
            seed=mc_seed,
        )
        if mc_result.mean_edge < kelly.ev_per_dollar - mc_ev_tolerance:
            return None

    return SideChoice(
        outcome=outcome,
        price=price,
        prob=kelly.prob,
        kelly=kelly,
        other_ev_per_dollar=other_ev,
        mc=mc_result,
    )


def _eligible(kelly: KellyResult, min_ev_threshold: float) -> bool:
    return (
        kelly.is_positive_ev
        and kelly.ev_per_dollar >= min_ev_threshold
        and kelly.position_usd >= 1.0
    )
