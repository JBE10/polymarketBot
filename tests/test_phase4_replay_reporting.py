"""
Phase 4 — decision snapshots, replay checks, and daily reporting.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.database import Database
from src.monitoring.daily_reporter import DailyReporter
from src.monitoring.replay import DecisionReplayService


@pytest.mark.asyncio
async def test_decision_snapshot_and_daily_aggregates(temp_db: Database) -> None:
    fixed_ts = "2026-04-11T12:00:00Z"
    await temp_db.insert_decision_snapshot(
        decision_ts=fixed_ts,
        asset="BTC",
        market_id="m1",
        side="UP",
        action="BUY",
        reason_code="SELECTED_TOP_N",
        regime="MID_VOL",
        p_model_up=0.60,
        p_market_up=0.50,
        edge_net_pct=7.0,
        score=4.0,
        threshold_pct=3.2,
        total_cost_pct=3.0,
        fee_pct=0.35,
        slippage_pct=2.5,
        latency_buffer_pct=0.15,
        notional_usd=20.0,
        spread_pct=0.8,
        depth_usd=3000.0,
        expiry_utc="2026-12-31T23:59:59Z",
        feature_ema_fast=101.0,
        feature_ema_slow=100.0,
        feature_rsi=56.0,
        feature_momentum=0.4,
        feature_atr_pctile=50.0,
        meta={"test": True},
    )

    await temp_db.insert_decision_snapshot(
        decision_ts=fixed_ts,
        asset="ETH",
        market_id="m2",
        side="DOWN",
        action="SKIP",
        reason_code="EDGE_BELOW_THRESHOLD",
        regime="HIGH_VOL",
        p_model_up=0.40,
        p_market_up=0.45,
        edge_net_pct=-2.0,
        score=0.05,
        threshold_pct=3.5,
        total_cost_pct=1.5,
        fee_pct=0.35,
        slippage_pct=1.0,
        latency_buffer_pct=0.15,
        notional_usd=0.0,
        spread_pct=1.1,
        depth_usd=2600.0,
        expiry_utc="2026-12-31T23:59:59Z",
        feature_ema_fast=100.0,
        feature_ema_slow=101.0,
        feature_rsi=45.0,
        feature_momentum=-0.2,
        feature_atr_pctile=75.0,
        meta={"test": True},
    )

    rows = await temp_db.get_decision_snapshots(limit=10)
    assert len(rows) == 2

    report = await temp_db.get_daily_decision_report(date_utc="2026-04-11")
    assert "by_asset" in report
    assert "by_regime" in report
    assert "by_hour" in report
    assert "by_side" in report
    assert len(report["by_asset"]) >= 1


@pytest.mark.asyncio
async def test_replay_detects_mismatch(temp_db: Database) -> None:
    # Expected net edge for this row is ((0.60 - 0.50) - 0.03) * 100 = 7.0
    # Intentionally store wrong edge to validate mismatch detection.
    await temp_db.insert_decision_snapshot(
        decision_ts="2026-04-11T12:30:00Z",
        asset="BTC",
        market_id="m3",
        side="UP",
        action="BUY",
        reason_code="SELECTED_TOP_N",
        regime="MID_VOL",
        p_model_up=0.60,
        p_market_up=0.50,
        edge_net_pct=6.0,
        score=4.0,
        threshold_pct=3.2,
        total_cost_pct=3.0,
        fee_pct=0.35,
        slippage_pct=2.5,
        latency_buffer_pct=0.15,
        notional_usd=20.0,
        spread_pct=0.8,
        depth_usd=3000.0,
        expiry_utc="2026-12-31T23:59:59Z",
        feature_ema_fast=101.0,
        feature_ema_slow=100.0,
        feature_rsi=56.0,
        feature_momentum=0.4,
        feature_atr_pctile=50.0,
        meta={"test": "mismatch"},
    )

    replayer = DecisionReplayService(temp_db)
    report = await replayer.replay_day(date_utc="2026-04-11")

    assert report["summary"]["total"] >= 1
    assert report["summary"]["mismatched"] >= 1


@pytest.mark.asyncio
async def test_daily_reporter_exports_csv(temp_db: Database, tmp_path: Path) -> None:
    await temp_db.insert_decision_snapshot(
        decision_ts="2026-04-11T14:00:00Z",
        asset="SOL",
        market_id="m4",
        side="UP",
        action="BUY",
        reason_code="SELECTED_TOP_N",
        regime="LOW_VOL",
        p_model_up=0.59,
        p_market_up=0.52,
        edge_net_pct=4.5,
        score=1.2,
        threshold_pct=4.8,
        total_cost_pct=2.5,
        fee_pct=0.35,
        slippage_pct=2.0,
        latency_buffer_pct=0.15,
        notional_usd=15.0,
        spread_pct=1.0,
        depth_usd=2200.0,
        expiry_utc="2026-12-31T23:59:59Z",
        feature_ema_fast=102.0,
        feature_ema_slow=100.0,
        feature_rsi=57.0,
        feature_momentum=0.2,
        feature_atr_pctile=20.0,
        meta={"test": "report"},
    )

    reporter = DailyReporter(db=temp_db, output_dir=str(tmp_path))
    report = await reporter.build_daily_report(date_utc="2026-04-11")

    target_date = report["date_utc"][0]["date"]
    assert (tmp_path / f"daily_report_asset_{target_date}.csv").exists()
    assert (tmp_path / f"daily_report_regime_{target_date}.csv").exists()
    assert (tmp_path / f"daily_report_hour_{target_date}.csv").exists()
    assert (tmp_path / f"daily_report_side_{target_date}.csv").exists()
