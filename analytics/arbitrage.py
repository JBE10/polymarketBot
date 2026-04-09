"""
Cross-platform arbitrage detection: Polymarket ↔ Kalshi.
Uses fuzzy text similarity to match markets across platforms.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from config import ARB_MIN_SPREAD_PCT, SIMILARITY_THRESHOLD


def find_arbitrage_opportunities(
    poly_markets: list[dict[str, Any]],
    kalshi_markets: list[dict[str, Any]],
    min_spread_pct: float = ARB_MIN_SPREAD_PCT,
    sim_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Pair Polymarket and Kalshi markets by question similarity, then surface
    any pairs where the absolute YES-price spread exceeds `min_spread_pct` %.

    Returns a list of opportunity dicts sorted by spread descending.
    """
    opportunities: list[dict[str, Any]] = []

    for pm in poly_markets:
        pm_q      = _normalise(pm.get("question", ""))
        pm_price  = _poly_yes_price(pm)
        if pm_price is None:
            continue

        best_match: dict[str, Any] | None = None
        best_score = 0.0

        for km in kalshi_markets:
            km_q = _normalise(
                km.get("title") or km.get("subtitle") or km.get("ticker", "")
            )
            score = _similarity(pm_q, km_q)
            if score > best_score and score >= sim_threshold:
                best_score = score
                best_match = km

        if best_match is None:
            continue

        km_price = _kalshi_yes_price(best_match)
        if km_price is None:
            continue

        spread     = pm_price - km_price          # positive = PM is more expensive
        spread_pct = abs(spread) * 100

        if spread_pct < min_spread_pct:
            continue

        buy_on  = "Kalshi"     if spread > 0 else "Polymarket"
        sell_on = "Polymarket" if spread > 0 else "Kalshi"

        opportunities.append(
            {
                "poly_question":  pm.get("question", "")[:70],
                "kalshi_title":   (
                    best_match.get("title")
                    or best_match.get("subtitle")
                    or best_match.get("ticker", "")
                )[:70],
                "similarity_pct": round(best_score * 100, 1),
                "poly_yes":       round(pm_price, 4),
                "kalshi_yes":     round(km_price, 4),
                "spread":         round(spread, 4),
                "spread_pct":     round(spread_pct, 2),
                "buy_on":         buy_on,
                "sell_on":        sell_on,
                "poly_market_id": pm.get("id", ""),
                "kalshi_ticker":  best_match.get("ticker", ""),
                "poly_volume":    float(pm.get("volume24hr", 0) or 0),
                "kalshi_volume":  float(best_match.get("volume", 0) or 0),
            }
        )

    return sorted(opportunities, key=lambda x: x["spread_pct"], reverse=True)


# ── Price extraction helpers ──────────────────────────────────────────────────

def _poly_yes_price(market: dict[str, Any]) -> float | None:
    """Extract the YES outcome price (0–1) from a Polymarket market dict."""
    try:
        prices   = market.get("outcomePrices") or []
        outcomes = market.get("outcomes") or []

        if prices and outcomes:
            for i, outcome in enumerate(outcomes):
                if str(outcome).lower() == "yes" and i < len(prices):
                    return float(prices[i])

        # Fall back to first price if no 'Yes' label
        if prices:
            return float(prices[0])
    except (ValueError, TypeError, IndexError):
        pass
    return None


def _kalshi_yes_price(market: dict[str, Any]) -> float | None:
    """
    Extract the YES price from a Kalshi market dict.
    Kalshi stores prices as integers in cents (0–99); divide by 100.
    """
    for field in ("yes_bid", "yes_ask", "last_price"):
        val = market.get(field)
        if val is not None:
            try:
                return float(val) / 100.0
            except (ValueError, TypeError):
                continue
    return None


# ── Text normalisation / similarity ──────────────────────────────────────────

_STOP = re.compile(
    r"\b(will|the|a|an|in|on|at|by|for|of|to|be|is|are|was|were|it|its|"
    r"this|that|with|from|or|and|whether)\b",
    re.IGNORECASE,
)
_NONALPHA = re.compile(r"[^a-z0-9 ]")


def _normalise(text: str) -> str:
    text = text.lower()
    text = _STOP.sub(" ", text)
    text = _NONALPHA.sub(" ", text)
    return " ".join(text.split())


def _similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher ratio after normalisation."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()
