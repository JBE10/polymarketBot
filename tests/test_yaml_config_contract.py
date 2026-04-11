from __future__ import annotations

from pathlib import Path

import pytest

from src.core.yaml_config import StrategyConfig


def _write_yaml(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / "strategy_config.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def test_load_demo_profile_contract(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
environment: demo

general:
  timezone: UTC
  candle_timeframes: ["1m", "5m"]
  decision_interval_seconds: 30
  pool_timeframe_minutes: 15
  no_trade_last_minutes_before_expiry: 3

universe:
  enabled_assets: ["BTC", "ETH"]
  max_assets_per_round: 2
  min_minutes_between_trades_same_asset: 15

data_sources:
  polymarket:
    use_api: true
    clob_base_url: "https://clob.polymarket.com"
    timeout_seconds: 8
    retries: 3
    retry_backoff_seconds: 1.2
  market_data:
    provider: "binance"
    timeout_seconds: 6
    retries: 3

features:
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  atr_period: 14
  momentum_lookback_bars_5m: 3
  volatility_lookback_bars_5m: 24

model:
  weights:
    ema_signal: 0.35
    rsi_signal: 0.20
    momentum: 0.30
    vol_regime: 0.15
  logistic_temperature: 1.0
  calibration:
    enabled: false

market_filters:
  min_orderbook_depth_usd:
    BTC: 3000
    ETH: 2500
  max_spread_pct:
    BTC: 1.0
    ETH: 1.2
  max_estimated_slippage_pct:
    BTC: 0.7
    ETH: 0.8

cost_model:
  taker_fee_pct: 0.20
  maker_fee_pct: 0.00
  use_taker_by_default: true
  extra_buffer_pct: 0.15

entry_rules:
  edge_threshold_pct:
    BTC: 3.2
    ETH: 3.5
  require_trend_alignment: true
  block_if_high_noise: true

ranking:
  enabled: true
  score_formula: "edge_net * liquidity_score * trend_quality * (1 - spread_penalty)"
  min_score_to_trade: 0.15

risk:
  bankroll_usd: 1000
  risk_per_trade_pct: 0.5
  max_total_exposure_pct: 2.5
  max_exposure_per_asset_pct: 1.0
  max_daily_drawdown_pct: 3.0
  consecutive_losses_cooldown:
    enabled: true
    losses: 2

position_sizing:
  method: "vol_adjusted_edge_scaled"
  target_volatility_reference: 1.0
  min_notional_usd: 15
  max_notional_usd_per_trade: 80
  edge_scale_cap: 2.0

execution:
  order_type_preference: "limit"
  limit_offset_ticks: 1
  time_in_force_seconds: 20
  cancel_if_not_filled: true
  max_requotes: 2
  idempotency_enabled: true

safety:
  kill_switch:
    on_api_error_streak: 6
    on_rejected_orders_streak: 4
    on_daily_dd_breach: true
  trading_hours_utc:
    enabled: false
    start: "00:00"
    end: "23:59"

logging:
  level: "INFO"
  structured_json: true
  save_decisions: true
  save_orderbook_snapshots: false
  metrics_output: "data/metrics.csv"
  trades_output: "data/trades.csv"

profiles:
  demo:
    risk:
      bankroll_usd: 1000
      risk_per_trade_pct: 0.5
      max_total_exposure_pct: 2.5
      max_exposure_per_asset_pct: 1.0
      max_daily_drawdown_pct: 3.0
      consecutive_losses_cooldown:
        enabled: true
        losses: 2
    entry_rules:
      edge_threshold_pct:
        BTC: 3.2
        ETH: 3.5
  real:
    risk:
      bankroll_usd: 1000
      risk_per_trade_pct: 0.3
      max_total_exposure_pct: 1.8
      max_exposure_per_asset_pct: 1.0
      max_daily_drawdown_pct: 2.2
      consecutive_losses_cooldown:
        enabled: true
        losses: 2
    market_filters:
      max_spread_pct:
        BTC: 0.9
        ETH: 1.0
    entry_rules:
      edge_threshold_pct:
        BTC: 3.8
        ETH: 4.0
""",
    )

    loaded = StrategyConfig.load(str(cfg))

    assert loaded.environment == "demo"
    assert loaded.ranking.min_score_to_trade == pytest.approx(0.15)
    assert loaded.execution.max_requotes == 2
    assert loaded.data_sources.polymarket.use_api is True


def test_real_profile_overrides_risk_values(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
environment: real

general:
  timezone: UTC
  candle_timeframes: ["1m", "5m"]
  decision_interval_seconds: 30
  pool_timeframe_minutes: 15
  no_trade_last_minutes_before_expiry: 3

universe:
  enabled_assets: ["BTC"]
  max_assets_per_round: 1
  min_minutes_between_trades_same_asset: 15

features:
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  atr_period: 14
  momentum_lookback_bars_5m: 3
  volatility_lookback_bars_5m: 24

market_filters:
  min_orderbook_depth_usd:
    BTC: 3000
  max_spread_pct:
    BTC: 1.0
  max_estimated_slippage_pct:
    BTC: 0.7

model:
  weights:
    ema_signal: 0.35
    rsi_signal: 0.20
    momentum: 0.30
    vol_regime: 0.15
  logistic_temperature: 1.0
  calibration:
    enabled: false

cost_model:
  taker_fee_pct: 0.20
  maker_fee_pct: 0.00
  use_taker_by_default: true
  extra_buffer_pct: 0.15

entry_rules:
  edge_threshold_pct:
    BTC: 3.2
  require_trend_alignment: true
  block_if_high_noise: true

risk:
  bankroll_usd: 1000
  risk_per_trade_pct: 0.5
  max_total_exposure_pct: 2.5
  max_exposure_per_asset_pct: 1.0
  max_daily_drawdown_pct: 3.0
  consecutive_losses_cooldown:
    enabled: true
    losses: 2

position_sizing:
  method: "vol_adjusted_edge_scaled"
  target_volatility_reference: 1.0
  min_notional_usd: 15
  max_notional_usd_per_trade: 80
  edge_scale_cap: 2.0

profiles:
  demo:
    risk:
      risk_per_trade_pct: 0.5
      max_total_exposure_pct: 2.5
      max_daily_drawdown_pct: 3.0
  real:
    risk:
      risk_per_trade_pct: 0.3
      max_total_exposure_pct: 1.8
      max_daily_drawdown_pct: 2.2
""",
    )

    loaded = StrategyConfig.load(str(cfg))

    assert loaded.environment == "real"
    assert loaded.risk.risk_per_trade_pct == pytest.approx(0.3)
    assert loaded.risk.max_total_exposure_pct == pytest.approx(1.8)


def test_missing_asset_threshold_is_rejected(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
environment: demo

general:
  timezone: UTC
  candle_timeframes: ["1m", "5m"]
  decision_interval_seconds: 30
  pool_timeframe_minutes: 15
  no_trade_last_minutes_before_expiry: 3

universe:
  enabled_assets: ["BTC", "ETH"]
  max_assets_per_round: 2
  min_minutes_between_trades_same_asset: 15

features:
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  atr_period: 14
  momentum_lookback_bars_5m: 3
  volatility_lookback_bars_5m: 24

market_filters:
  min_orderbook_depth_usd:
    BTC: 3000
    ETH: 2500
  max_spread_pct:
    BTC: 1.0
    ETH: 1.2
  max_estimated_slippage_pct:
    BTC: 0.7
    ETH: 0.8

model:
  weights:
    ema_signal: 0.35
    rsi_signal: 0.20
    momentum: 0.30
    vol_regime: 0.15
  logistic_temperature: 1.0
  calibration:
    enabled: false

cost_model:
  taker_fee_pct: 0.20
  maker_fee_pct: 0.00
  use_taker_by_default: true
  extra_buffer_pct: 0.15

entry_rules:
  edge_threshold_pct:
    BTC: 3.2
  require_trend_alignment: true
  block_if_high_noise: true

risk:
  bankroll_usd: 1000
  risk_per_trade_pct: 0.5
  max_total_exposure_pct: 2.5
  max_exposure_per_asset_pct: 1.0
  max_daily_drawdown_pct: 3.0
  consecutive_losses_cooldown:
    enabled: true
    losses: 2

position_sizing:
  method: "vol_adjusted_edge_scaled"
  target_volatility_reference: 1.0
  min_notional_usd: 15
  max_notional_usd_per_trade: 80
  edge_scale_cap: 2.0

profiles:
  demo:
    risk:
      risk_per_trade_pct: 0.5
      max_total_exposure_pct: 2.5
      max_daily_drawdown_pct: 3.0
""",
    )

    with pytest.raises(ValueError, match="missing enabled assets"):
        StrategyConfig.load(str(cfg))


def test_real_risk_cannot_be_looser_than_demo(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
environment: demo

general:
  timezone: UTC
  candle_timeframes: ["1m", "5m"]
  decision_interval_seconds: 30
  pool_timeframe_minutes: 15
  no_trade_last_minutes_before_expiry: 3

universe:
  enabled_assets: ["BTC"]
  max_assets_per_round: 1
  min_minutes_between_trades_same_asset: 15

features:
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  atr_period: 14
  momentum_lookback_bars_5m: 3
  volatility_lookback_bars_5m: 24

market_filters:
  min_orderbook_depth_usd:
    BTC: 3000
  max_spread_pct:
    BTC: 1.0
  max_estimated_slippage_pct:
    BTC: 0.7

model:
  weights:
    ema_signal: 0.35
    rsi_signal: 0.20
    momentum: 0.30
    vol_regime: 0.15
  logistic_temperature: 1.0
  calibration:
    enabled: false

cost_model:
  taker_fee_pct: 0.20
  maker_fee_pct: 0.00
  use_taker_by_default: true
  extra_buffer_pct: 0.15

entry_rules:
  edge_threshold_pct:
    BTC: 3.2
  require_trend_alignment: true
  block_if_high_noise: true

risk:
  bankroll_usd: 1000
  risk_per_trade_pct: 0.5
  max_total_exposure_pct: 2.5
  max_exposure_per_asset_pct: 1.0
  max_daily_drawdown_pct: 3.0
  consecutive_losses_cooldown:
    enabled: true
    losses: 2

position_sizing:
  method: "vol_adjusted_edge_scaled"
  target_volatility_reference: 1.0
  min_notional_usd: 15
  max_notional_usd_per_trade: 80
  edge_scale_cap: 2.0

profiles:
  demo:
    risk:
      risk_per_trade_pct: 0.5
      max_total_exposure_pct: 2.5
      max_daily_drawdown_pct: 3.0
  real:
    risk:
      risk_per_trade_pct: 0.8
      max_total_exposure_pct: 2.6
      max_daily_drawdown_pct: 3.1
""",
    )

    with pytest.raises(ValueError, match="real risk_per_trade_pct must be <= demo"):
        StrategyConfig.load(str(cfg))
