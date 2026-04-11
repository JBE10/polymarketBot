from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.yaml_config import StrategyConfig
from src.orchestrator.quant_engine import QuantEngine
from src.storage.models import Features, MarketState


class _FakeDB:
    async def get_daily_mm_pnl(self) -> float:
        return 0.0

    async def get_open_positions(self):
        return []

    async def get_consecutive_losses(self, _asset_symbol: str) -> int:
        return 0


class _CustomFeed:
    def __init__(self, features: Features) -> None:
        self._features = features

    def get_features(self, _asset: str) -> Features:
        return self._features


@pytest.mark.asyncio
async def test_blocks_market_inside_no_trade_window() -> None:
    cfg = StrategyConfig.load("strategy_config.yaml")
    engine = QuantEngine(db=_FakeDB(), config=cfg)

    market = MarketState(
        asset="BTC",
        market_id="m1",
        expiry_utc=datetime.now(timezone.utc) + timedelta(minutes=2),
        p_market_up=0.40,
        spread_pct=0.5,
        depth_usd=5000,
        best_bid=0.39,
        best_ask=0.41,
    )

    decisions = await engine.evaluate_round([market])

    assert decisions == []


@pytest.mark.asyncio
async def test_ranking_score_uses_edge_percentage_points() -> None:
    cfg = StrategyConfig.load("strategy_config.yaml")
    cfg.position_sizing.min_notional_usd = 1.0
    engine = QuantEngine(db=_FakeDB(), config=cfg)

    market = MarketState(
        asset="BTC",
        market_id="m2",
        expiry_utc=datetime.now(timezone.utc) + timedelta(minutes=10),
        p_market_up=0.40,
        spread_pct=0.5,
        depth_usd=5000,
        best_bid=0.39,
        best_ask=0.41,
    )

    decisions = await engine.evaluate_round([market])

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.asset == "BTC"
    assert decision.edge_net_pct > 0
    assert decision.score >= cfg.ranking.min_score_to_trade


@pytest.mark.asyncio
async def test_blocks_when_atr_percentile_below_noise_floor() -> None:
    cfg = StrategyConfig.load("strategy_config.yaml")
    cfg.position_sizing.min_notional_usd = 1.0
    cfg.entry_rules.block_if_high_noise = True
    cfg.entry_rules.high_noise_rules = {"min_atr_percentile": 60}
    engine = QuantEngine(db=_FakeDB(), config=cfg)
    engine.price_feed = _CustomFeed(
        Features(
            ema_fast=101.0,
            ema_slow=100.0,
            rsi=56.0,
            momentum=0.4,
            atr_pctile=40.0,
        )
    )

    market = MarketState(
        asset="BTC",
        market_id="m3",
        expiry_utc=datetime.now(timezone.utc) + timedelta(minutes=10),
        p_market_up=0.40,
        spread_pct=0.5,
        depth_usd=5000,
        best_bid=0.39,
        best_ask=0.41,
    )

    decisions = await engine.evaluate_round([market])

    assert decisions == []


@pytest.mark.asyncio
async def test_allows_trade_when_noise_filter_disabled() -> None:
    cfg = StrategyConfig.load("strategy_config.yaml")
    cfg.position_sizing.min_notional_usd = 1.0
    cfg.entry_rules.block_if_high_noise = False
    cfg.entry_rules.high_noise_rules = {"min_atr_percentile": 95}
    engine = QuantEngine(db=_FakeDB(), config=cfg)
    engine.price_feed = _CustomFeed(
        Features(
            ema_fast=101.0,
            ema_slow=100.0,
            rsi=56.0,
            momentum=0.4,
            atr_pctile=20.0,
        )
    )

    market = MarketState(
        asset="BTC",
        market_id="m4",
        expiry_utc=datetime.now(timezone.utc) + timedelta(minutes=10),
        p_market_up=0.40,
        spread_pct=0.5,
        depth_usd=5000,
        best_bid=0.39,
        best_ask=0.41,
    )

    decisions = await engine.evaluate_round([market])

    assert len(decisions) == 1
