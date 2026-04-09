"""
Market risk scoring.

Composite score 0–100 (lower = less risky) built from five sub-scores:
  • liquidity  — depth of the order book / liquidity pool
  • activity   — recent trading volume relative to liquidity
  • time       — days remaining until resolution
  • certainty  — how far the YES price is from 50 % (extreme = more certain)
  • age        — how long the market has been live (older = more data)

Weights are intentionally biased toward liquidity and activity since those
are the most actionable signals for a trader.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

_WEIGHTS = {
    "liquidity": 0.35,
    "activity":  0.25,
    "time":      0.20,
    "certainty": 0.10,
    "age":       0.10,
}


def score_market(market: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the full risk profile for a single market dict.
    Returns a dict with composite_score, risk_level, and per-dimension breakdown.
    """
    liquidity = float(
        market.get("liquidityNum") or market.get("liquidity") or 0
    )
    volume_24h = float(market.get("volume24hr") or 0)
    end_date   = market.get("endDateIso") or market.get("endDate")
    start_date = market.get("startDateIso") or market.get("startDate")
    yes_price  = _yes_price(market)

    breakdown = {
        "liquidity": _score_liquidity(liquidity),
        "activity":  _score_activity(volume_24h, liquidity),
        "time":      _score_time(end_date),
        "certainty": _score_certainty(yes_price),
        "age":       _score_age(start_date),
    }

    composite = sum(breakdown[k] * _WEIGHTS[k] for k in breakdown)

    risk_level = (
        "LOW"    if composite < 30 else
        "MEDIUM" if composite < 65 else
        "HIGH"
    )

    return {
        "composite_score": round(composite, 1),
        "risk_level":      risk_level,
        "breakdown":       {k: round(v, 1) for k, v in breakdown.items()},
        "liquidity_usd":   round(liquidity, 0),
        "volume_24h_usd":  round(volume_24h, 0),
        "yes_price":       yes_price,
    }


def rank_by_risk(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach '_risk' sub-dict to each market and sort ascending by composite score."""
    ranked = []
    for m in markets:
        entry = dict(m)
        entry["_risk"] = score_market(m)
        ranked.append(entry)
    return sorted(ranked, key=lambda x: x["_risk"]["composite_score"])


# ── Sub-score functions (all return 0–100) ────────────────────────────────────

def _score_liquidity(usd: float) -> float:
    """Low liquidity → high risk.  ≥ $500k = near-zero risk."""
    if usd <= 0:
        return 100.0
    if usd >= 500_000:
        return 5.0
    return max(5.0, 100.0 - (math.log10(usd) / math.log10(500_000)) * 95.0)


def _score_activity(vol_24h: float, liquidity: float) -> float:
    """Inactive market relative to its size = higher uncertainty."""
    if liquidity <= 0:
        return 100.0
    if vol_24h <= 0:
        return 85.0
    ratio = vol_24h / liquidity
    if   ratio >= 0.10: return 10.0
    elif ratio >= 0.01: return 30.0
    elif ratio >= 0.001: return 55.0
    return 78.0


def _score_time(end_date_str: str | None) -> float:
    """More days remaining → more unresolved uncertainty."""
    if not end_date_str:
        return 50.0
    try:
        end_dt = _parse_iso(end_date_str)
        days   = (end_dt - _now()).days
        if   days <= 0:   return 5.0   # already resolved / imminent
        elif days <= 7:   return 15.0
        elif days <= 30:  return 35.0
        elif days <= 90:  return 52.0
        elif days <= 365: return 68.0
        return 85.0
    except Exception:
        return 50.0


def _score_certainty(yes_price: float | None) -> float:
    """
    Price at 50 % → maximum uncertainty (score 100).
    Price near 0 or 100 % → market is almost decided (score near 0).
    """
    if yes_price is None:
        return 50.0
    return round(100.0 * (1.0 - abs(yes_price - 0.5) * 2.0), 1)


def _score_age(start_date_str: str | None) -> float:
    """Very new markets have little trading history → higher model uncertainty."""
    if not start_date_str:
        return 50.0
    try:
        start_dt = _parse_iso(start_date_str)
        days     = (_now() - start_dt).days
        if   days <= 1:  return 88.0
        elif days <= 7:  return 60.0
        elif days <= 30: return 35.0
        return 15.0
    except Exception:
        return 50.0


# ── Utility ───────────────────────────────────────────────────────────────────

def _yes_price(market: dict[str, Any]) -> float | None:
    try:
        prices   = market.get("outcomePrices") or []
        outcomes = market.get("outcomes") or []
        if prices and outcomes:
            for i, o in enumerate(outcomes):
                if str(o).lower() == "yes" and i < len(prices):
                    return float(prices[i])
        if prices:
            return float(prices[0])
    except (ValueError, TypeError, IndexError):
        pass
    return None


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now() -> datetime:
    return datetime.now(timezone.utc)
