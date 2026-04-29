"""
Kelly Criterion and Expected Value mathematics for binary prediction markets.

Background
----------
A binary prediction market lets you buy a YES share at price p ∈ (0, 1).
If YES resolves:  you receive $1 per share → net profit = (1 - p) per dollar staked
If NO  resolves:  share expires worthless → net loss = p per dollar staked

In Kelly notation:
    b = net odds on a win = (1 - p) / p     (how much you win per dollar risked)
    f* = (b · prob - (1 - prob)) / b        (full Kelly fraction of bankroll)
    fractional Kelly = f* × scale           (scaled to reduce variance)

Expected Value per dollar of stake:
    EV = prob - p                           (positive = edge in your favour)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyResult:
    """All computed sizing metrics for a single market opportunity."""

    prob:             float   # estimated P(YES)
    entry_price:      float   # current market price (YES)
    full_kelly:       float   # unconstrained Kelly fraction
    frac_kelly:       float   # fractional Kelly (scaled)
    ev_per_dollar:    float   # expected value per $ staked
    edge_pct:         float   # EV as a percentage of entry price
    position_usd:     float   # dollar amount to wager given bankroll
    shares:           float   # number of YES shares at position_usd / entry_price
    expected_profit:  float   # position_usd × ev_per_dollar
    is_positive_ev:   bool

    def __str__(self) -> str:
        return (
            f"KellyResult("
            f"prob={self.prob:.3f}, price={self.entry_price:.3f}, "
            f"EV={self.ev_per_dollar:+.4f} ({self.edge_pct:+.1f}%), "
            f"frac_kelly={self.frac_kelly:.4f}, "
            f"position=${self.position_usd:.2f}, "
            f"{'✓ BUY' if self.is_positive_ev else '✗ SKIP'}"
            f")"
        )


def kelly_criterion(
    prob: float,
    entry_price: float,
    kelly_fraction: float = 0.25,
    bankroll: float = 0.0,
    max_position_usd: float = 0.0,
) -> float:
    """
    Return the fractional Kelly wager as a fraction of bankroll (0–1).

    Parameters
    ----------
    prob              : estimated probability of YES resolving
    entry_price       : current YES price in the market (0 < p < 1)
    kelly_fraction    : scale factor applied to full Kelly (0.25 = quarter-Kelly)
    bankroll          : total available capital (only used if > 0)
    max_position_usd  : hard cap on position size in USD (only used if > 0)

    Returns the fraction of bankroll to wager (0.0 if no edge).
    """
    if not (0 < entry_price < 1) or not (0 < prob < 1):
        return 0.0

    # Net odds: winning (1-p) per dollar staked vs losing p per dollar
    b = (1.0 - entry_price) / entry_price
    q = 1.0 - prob

    full_kelly = (b * prob - q) / b  # (bp - q) / b

    if full_kelly <= 0:
        return 0.0

    frac = min(full_kelly * kelly_fraction, 0.20)  # hard cap at 20% of bankroll
    return frac


def compute_kelly(
    prob: float,
    entry_price: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_position_usd: float = 100.0,
) -> KellyResult:
    """
    Full computation returning a KellyResult dataclass with all metrics.

    This is the primary entry-point used by the strategy engine.
    """
    if not (0 < entry_price < 1):
        raise ValueError(f"entry_price must be in (0, 1), got {entry_price}")
    if not (0 <= prob <= 1):
        raise ValueError(f"prob must be in [0, 1], got {prob}")

    b = (1.0 - entry_price) / entry_price
    q = 1.0 - prob

    full_k = (b * prob - q) / b if b > 0 else 0.0
    frac_k = max(0.0, min(full_k * kelly_fraction, 0.20))

    ev    = prob - entry_price          # EV per dollar staked
    edge  = ev / entry_price * 100      # edge as % of cost

    # Position sizing
    position = frac_k * bankroll
    if max_position_usd > 0:
        position = min(position, max_position_usd)
    if position < 0:
        position = 0.0

    shares = position / entry_price if entry_price > 0 else 0.0
    exp_profit = position * ev

    return KellyResult(
        prob=round(prob, 6),
        entry_price=round(entry_price, 6),
        full_kelly=round(full_k, 6),
        frac_kelly=round(frac_k, 6),
        ev_per_dollar=round(ev, 6),
        edge_pct=round(edge, 3),
        position_usd=round(position, 4),
        shares=round(shares, 4),
        expected_profit=round(exp_profit, 4),
        is_positive_ev=ev > 0 and frac_k > 0,
    )


def expected_value(prob: float, entry_price: float) -> float:
    """
    EV per dollar staked.  Positive = edge in our favour.

    EV = p × (1 - price) - (1 - p) × price = p - price
    """
    return prob - entry_price


def min_prob_for_positive_ev(entry_price: float) -> float:
    """Minimum probability required for any edge (breakeven probability)."""
    return entry_price


def max_loss(position_usd: float) -> float:
    """Worst-case loss: entire stake if YES does not resolve."""
    return position_usd


def multi_market_kelly(
    opportunities: list[tuple[float, float]],
    bankroll: float,
    kelly_fraction: float = 0.25,
) -> list[float]:
    """
    Naive simultaneous Kelly for multiple independent bets.
    Each tuple is (prob, entry_price).  Returns a list of dollar amounts.

    Note: True simultaneous Kelly requires solving a system of equations;
    this is a practical approximation that treats each bet independently
    then normalises if total exceeds bankroll.
    """
    sizes = []
    for prob, price in opportunities:
        frac = kelly_criterion(prob, price, kelly_fraction)
        sizes.append(frac * bankroll)

    total = sum(sizes)
    if total > bankroll:
        scale = bankroll / total
        sizes = [s * scale for s in sizes]

    return sizes
