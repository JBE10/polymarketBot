from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.storage.database import Database


@dataclass
class PromotionCriteria:
    min_trades: int = 150
    min_profit_factor: float = 1.15
    min_ev_per_trade: float = 0.0
    max_drawdown_pct: float = 2.2
    max_severe_incidents: int = 0


class PromotionGateEvaluator:
    """Evaluates Demo -> Real promotion criteria from persisted paper-trading data."""

    def __init__(self, db: Database, *, bankroll_usd: float) -> None:
        self._db = db
        self._bankroll_usd = bankroll_usd

    @staticmethod
    def _profit_factor(gross_profit: float, gross_loss: float) -> float:
        if gross_loss <= 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _max_drawdown_pct(self, pnl_values: list[float]) -> float:
        if not pnl_values or self._bankroll_usd <= 0:
            return 0.0

        equity = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnl_values:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        return (max_dd / self._bankroll_usd) * 100.0

    async def evaluate(
        self,
        *,
        criteria: PromotionCriteria,
        start_ts: str | None = None,
        end_ts: str | None = None,
    ) -> dict[str, Any]:
        perf = await self._db.get_mm_performance_summary(start_ts=start_ts, end_ts=end_ts)
        rounds = await self._db.get_mm_closed_rounds(start_ts=start_ts, end_ts=end_ts)
        severe_incidents = await self._db.get_operational_incidents(
            start_ts=start_ts,
            end_ts=end_ts,
            severity="SEVERE",
            limit=100000,
        )

        pnl_values = [float((r.get("realized_pnl") or 0.0) + (r.get("rebate_est") or 0.0)) for r in rounds]
        max_dd_pct = self._max_drawdown_pct(pnl_values)
        pf = self._profit_factor(float(perf["gross_profit"]), float(perf["gross_loss"]))

        checks = {
            "trades": int(perf["trades"]) >= criteria.min_trades,
            "ev_per_trade": float(perf["ev_per_trade"]) > criteria.min_ev_per_trade,
            "profit_factor": pf > criteria.min_profit_factor,
            "max_drawdown_pct": max_dd_pct <= criteria.max_drawdown_pct,
            "severe_incidents": len(severe_incidents) <= criteria.max_severe_incidents,
        }

        return {
            "window": {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "metrics": {
                "trades": int(perf["trades"]),
                "wins": int(perf["wins"]),
                "losses": int(perf["losses"]),
                "gross_profit": float(perf["gross_profit"]),
                "gross_loss": float(perf["gross_loss"]),
                "net_pnl": float(perf["net_pnl"]),
                "ev_per_trade": float(perf["ev_per_trade"]),
                "profit_factor": pf,
                "max_drawdown_pct": max_dd_pct,
                "severe_incidents": len(severe_incidents),
            },
            "criteria": {
                "min_trades": criteria.min_trades,
                "min_profit_factor": criteria.min_profit_factor,
                "min_ev_per_trade": criteria.min_ev_per_trade,
                "max_drawdown_pct": criteria.max_drawdown_pct,
                "max_severe_incidents": criteria.max_severe_incidents,
            },
            "checks": checks,
            "approved": all(checks.values()),
        }
