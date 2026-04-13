"""
Phase 5 — promotion gate evaluation for Demo -> Real.
"""
from __future__ import annotations

import pytest

from src.core.database import Database
from src.monitoring.promotion_gate import PromotionCriteria, PromotionGateEvaluator


@pytest.mark.asyncio
async def test_operational_incident_logging_and_filtering(temp_db: Database) -> None:
    await temp_db.insert_operational_incident(
        source="market_maker",
        severity="SEVERE",
        incident_type="kill_switch",
        message="api streak hit",
        details={"api_error_streak": 6},
    )
    await temp_db.insert_operational_incident(
        source="market_maker",
        severity="WARN",
        incident_type="retry",
        message="429 retry",
        details={"attempt": 2},
    )

    severe = await temp_db.get_operational_incidents(severity="SEVERE")
    warns = await temp_db.get_operational_incidents(severity="WARN")

    assert len(severe) == 1
    assert len(warns) == 1
    assert severe[0]["incident_type"] == "kill_switch"


@pytest.mark.asyncio
async def test_promotion_gate_approved_when_criteria_met(temp_db: Database) -> None:
    # 4 winning closed rounds, no incidents.
    for i in range(4):
        rid = await temp_db.insert_mm_round(
            market_id=f"m{i}",
            token_id=f"t{i}",
            question="q",
            buy_price=0.60,
            shares=100.0,
        )
        await temp_db.close_mm_round(
            rid,
            sell_price=0.62,
            realized_pnl=2.0,
            rebate_est=0.1,
        )

    evaluator = PromotionGateEvaluator(temp_db, bankroll_usd=1000.0)
    criteria = PromotionCriteria(
        min_trades=4,
        min_profit_factor=1.0,
        min_ev_per_trade=0.0,
        max_drawdown_pct=5.0,
        max_severe_incidents=0,
    )
    report = await evaluator.evaluate(criteria=criteria)

    assert report["approved"] is True
    assert report["metrics"]["trades"] == 4
    assert report["metrics"]["severe_incidents"] == 0


@pytest.mark.asyncio
async def test_promotion_gate_rejected_on_severe_incident(temp_db: Database) -> None:
    rid = await temp_db.insert_mm_round(
        market_id="mx",
        token_id="tx",
        question="q",
        buy_price=0.60,
        shares=100.0,
    )
    await temp_db.close_mm_round(
        rid,
        sell_price=0.62,
        realized_pnl=2.0,
        rebate_est=0.1,
    )
    await temp_db.insert_operational_incident(
        source="market_maker",
        severity="SEVERE",
        incident_type="kill_switch",
        message="rejected order streak",
        details={"rejected_order_streak": 4},
    )

    evaluator = PromotionGateEvaluator(temp_db, bankroll_usd=1000.0)
    criteria = PromotionCriteria(
        min_trades=1,
        min_profit_factor=1.0,
        min_ev_per_trade=0.0,
        max_drawdown_pct=10.0,
        max_severe_incidents=0,
    )
    report = await evaluator.evaluate(criteria=criteria)

    assert report["approved"] is False
    assert report["checks"]["severe_incidents"] is False
