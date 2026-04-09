"""
Whale activity detection.
Identifies large trades based on a configurable USD threshold.
"""
from __future__ import annotations

from typing import Any

from config import WHALE_THRESHOLD_USD


def detect_whale_trades(
    trades: list[dict[str, Any]],
    threshold_usd: float = WHALE_THRESHOLD_USD,
) -> list[dict[str, Any]]:
    """
    Filter a list of CLOB trades to those whose notional value meets the threshold.
    Returns records enriched with 'value_usd' and display-ready address snippets.
    """
    whales: list[dict[str, Any]] = []

    for trade in trades:
        try:
            price = float(trade.get("price", 0) or 0)
            size  = float(trade.get("size",  0) or 0)
            value = price * size

            if value < threshold_usd:
                continue

            maker = trade.get("makerAddress") or ""
            taker = trade.get("takerAddress") or ""
            tx    = trade.get("transactionHash") or ""

            whales.append(
                {
                    "market_id":    trade.get("_market_id", trade.get("market", "")),
                    "question":     trade.get("_question", ""),
                    "tx_hash":      (tx[:10] + "…") if tx else "N/A",
                    "side":         trade.get("side", "").upper() or "N/A",
                    "outcome":      _resolve_outcome(trade),
                    "price":        price,
                    "size":         size,
                    "value_usd":    round(value, 2),
                    "maker":        (_truncate_addr(maker)) if maker else "N/A",
                    "taker":        (_truncate_addr(taker)) if taker else "N/A",
                    "timestamp":    trade.get("timestamp", ""),
                    "fee_rate_bps": trade.get("feeRateBps", "N/A"),
                }
            )
        except (ValueError, TypeError):
            continue

    return sorted(whales, key=lambda x: x["value_usd"], reverse=True)


def whale_sentiment(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarise net buy/sell whale pressure for a set of trades.
    Returns sentiment label, buy %, sell %, and total whale notional.
    """
    whale_trades = detect_whale_trades(trades)

    buy_vol  = sum(t["value_usd"] for t in whale_trades if t["side"] == "BUY")
    sell_vol = sum(t["value_usd"] for t in whale_trades if t["side"] == "SELL")
    total    = buy_vol + sell_vol

    if total == 0:
        return {
            "sentiment": "NEUTRAL",
            "buy_pct": 50.0,
            "sell_pct": 50.0,
            "total_whale_usd": 0.0,
            "count": 0,
        }

    buy_pct  = buy_vol  / total * 100
    sell_pct = sell_vol / total * 100
    label    = "BULLISH" if buy_pct >= 65 else "BEARISH" if sell_pct >= 65 else "NEUTRAL"

    return {
        "sentiment": label,
        "buy_pct": round(buy_pct, 1),
        "sell_pct": round(sell_pct, 1),
        "total_whale_usd": round(total, 2),
        "count": len(whale_trades),
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_outcome(trade: dict[str, Any]) -> str:
    """Best-effort outcome label from trade metadata."""
    side = (trade.get("side") or "").upper()
    if side == "BUY":
        return "YES"
    if side == "SELL":
        return "NO"
    return trade.get("outcome", "?").upper()


def _truncate_addr(addr: str) -> str:
    """Return a compact address: 0x1234…abcd."""
    addr = addr.strip()
    if len(addr) > 12:
        return f"{addr[:6]}…{addr[-4:]}"
    return addr
