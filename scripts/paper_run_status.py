from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.database import Database
from src.core.yaml_config import StrategyConfig
from src.monitoring.promotion_gate import PromotionCriteria, PromotionGateEvaluator


async def main() -> None:
    cfg = get_settings()
    strategy_cfg = StrategyConfig.load("strategy_config.yaml")

    db = Database(cfg.db_path)
    await db.connect()

    try:
        evaluator = PromotionGateEvaluator(db, bankroll_usd=strategy_cfg.risk.bankroll_usd)
        criteria = PromotionCriteria(
            min_trades=150,
            min_profit_factor=1.15,
            min_ev_per_trade=0.0,
            max_drawdown_pct=2.2,
            max_severe_incidents=0,
        )

        report = await evaluator.evaluate(criteria=criteria)
        metrics = report["metrics"]

        trades = int(metrics["trades"])
        remaining = max(criteria.min_trades - trades, 0)

        summary = {
            "status": "approved" if report["approved"] else "not-approved",
            "trades_done": trades,
            "trades_remaining_to_goal": remaining,
            "ev_per_trade": metrics["ev_per_trade"],
            "profit_factor": metrics["profit_factor"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "severe_incidents": metrics["severe_incidents"],
            "checks": report["checks"],
        }

        print(json.dumps(summary, indent=2))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
