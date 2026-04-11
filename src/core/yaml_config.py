import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

def _deep_merge(target: Dict[Any, Any], source: Dict[Any, Any]) -> Dict[Any, Any]:
    """Recursively deep merges source into target."""
    for key, val in source.items():
        if isinstance(val, dict) and key in target and isinstance(target[key], dict):
            target[key] = _deep_merge(target[key], val)
        else:
            target[key] = val
    return target


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_per_asset_map(
    map_name: str,
    values: Dict[str, float],
    enabled_assets: List[str],
    *,
    min_value: float,
    max_value: float,
) -> None:
    missing = sorted([a for a in enabled_assets if a not in values])
    if missing:
        raise ValueError(f"{map_name} is missing enabled assets: {', '.join(missing)}")

    for asset in enabled_assets:
        v = values[asset]
        if not (min_value <= v <= max_value):
            raise ValueError(
                f"{map_name}[{asset}]={v} outside valid range [{min_value}, {max_value}]"
            )

# Pydantic Models for Mapping Config
class GeneralConfig(StrictBaseModel):
    timezone: str = "UTC"
    candle_timeframes: List[str] = ["1m", "5m"]
    decision_interval_seconds: int = 30
    pool_timeframe_minutes: int = 15
    no_trade_last_minutes_before_expiry: int = 3

    @field_validator("decision_interval_seconds", "pool_timeframe_minutes")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("no_trade_last_minutes_before_expiry")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

class UniverseConfig(StrictBaseModel):
    enabled_assets: List[str] = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    max_assets_per_round: int = 2
    min_minutes_between_trades_same_asset: int = 15

    @field_validator("enabled_assets")
    @classmethod
    def _validate_enabled_assets(cls, assets: List[str]) -> List[str]:
        normalized = [a.upper().strip() for a in assets]
        if not normalized:
            raise ValueError("enabled_assets cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("enabled_assets contains duplicates")
        return normalized

    @field_validator("max_assets_per_round")
    @classmethod
    def _validate_max_assets_per_round(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_assets_per_round must be > 0")
        return value

class FeaturesConfig(StrictBaseModel):
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    momentum_lookback_bars_5m: int = 3
    volatility_lookback_bars_5m: int = 24

    @model_validator(mode="after")
    def _validate_ema_order(self):
        if self.ema_fast <= 0 or self.ema_slow <= 0:
            raise ValueError("EMA periods must be > 0")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be < ema_slow")
        return self

class ModelWeightsConfig(StrictBaseModel):
    ema_signal: float = 0.35
    rsi_signal: float = 0.20
    momentum: float = 0.30
    vol_regime: float = 0.15

class CalibrationConfig(StrictBaseModel):
    enabled: bool = False

class ModelConfig(StrictBaseModel):
    weights: ModelWeightsConfig = Field(default_factory=ModelWeightsConfig)
    logistic_temperature: float = 1.0
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @field_validator("logistic_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("logistic_temperature must be > 0")
        return value

class MarketFiltersConfig(StrictBaseModel):
    min_orderbook_depth_usd: Dict[str, float] = Field(default_factory=dict)
    max_spread_pct: Dict[str, float] = Field(default_factory=dict)
    max_estimated_slippage_pct: Dict[str, float] = Field(default_factory=dict)

class EntryRulesConfig(StrictBaseModel):
    edge_threshold_pct: Dict[str, float] = Field(default_factory=dict)
    require_trend_alignment: bool = True
    block_if_high_noise: bool = True
    trend_rules: Dict[str, str] = Field(default_factory=dict)
    high_noise_rules: Dict[str, float] = Field(default_factory=dict)

class RiskConfig(StrictBaseModel):
    bankroll_usd: float = 1000.0
    risk_per_trade_pct: float = 0.50
    max_total_exposure_pct: float = 2.5
    max_exposure_per_asset_pct: float = 1.0
    max_daily_drawdown_pct: float = 3.0
    consecutive_losses_cooldown: Dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "risk_per_trade_pct",
        "max_total_exposure_pct",
        "max_exposure_per_asset_pct",
        "max_daily_drawdown_pct",
    )
    @classmethod
    def _validate_percent_positive(cls, value: float) -> float:
        if value <= 0 or value > 100:
            raise ValueError("percentage must be in (0, 100]")
        return value

    @model_validator(mode="after")
    def _validate_risk_hierarchy(self):
        if self.max_exposure_per_asset_pct > self.max_total_exposure_pct:
            raise ValueError("max_exposure_per_asset_pct cannot exceed max_total_exposure_pct")
        return self

class CostModelConfig(StrictBaseModel):
    taker_fee_pct: float = 0.20
    maker_fee_pct: float = 0.00
    use_taker_by_default: bool = True
    extra_buffer_pct: float = 0.15

    @field_validator("taker_fee_pct", "maker_fee_pct", "extra_buffer_pct")
    @classmethod
    def _validate_cost_percent(cls, value: float) -> float:
        if value < 0 or value > 100:
            raise ValueError("cost percentages must be in [0, 100]")
        return value

class PositionSizingConfig(StrictBaseModel):
    method: str = "vol_adjusted_edge_scaled"
    target_volatility_reference: float = 1.0
    min_notional_usd: float = 15.0
    max_notional_usd_per_trade: float = 80.0
    edge_scale_cap: float = 2.0

    @model_validator(mode="after")
    def _validate_sizing_bounds(self):
        if self.min_notional_usd < 0:
            raise ValueError("min_notional_usd must be >= 0")
        if self.max_notional_usd_per_trade <= 0:
            raise ValueError("max_notional_usd_per_trade must be > 0")
        if self.min_notional_usd > self.max_notional_usd_per_trade:
            raise ValueError("min_notional_usd cannot exceed max_notional_usd_per_trade")
        if self.edge_scale_cap <= 0:
            raise ValueError("edge_scale_cap must be > 0")
        return self


class PolymarketSourceConfig(StrictBaseModel):
    use_api: bool = True
    clob_base_url: str = "https://clob.polymarket.com"
    timeout_seconds: int = 8
    retries: int = 3
    retry_backoff_seconds: float = 1.2


class MarketDataSourceConfig(StrictBaseModel):
    provider: str = "binance"
    timeout_seconds: int = 6
    retries: int = 3


class DataSourcesConfig(StrictBaseModel):
    polymarket: PolymarketSourceConfig = Field(default_factory=PolymarketSourceConfig)
    market_data: MarketDataSourceConfig = Field(default_factory=MarketDataSourceConfig)


class RankingConfig(StrictBaseModel):
    enabled: bool = True
    score_formula: str = "edge_net * liquidity_score * trend_quality * (1 - spread_penalty)"
    min_score_to_trade: float = 0.15

    @field_validator("min_score_to_trade")
    @classmethod
    def _validate_min_score(cls, value: float) -> float:
        if value < 0:
            raise ValueError("min_score_to_trade must be >= 0")
        return value


class ExecutionConfig(StrictBaseModel):
    order_type_preference: str = "limit"
    limit_offset_ticks: int = 1
    time_in_force_seconds: int = 20
    cancel_if_not_filled: bool = True
    max_requotes: int = 2
    idempotency_enabled: bool = True


class KillSwitchConfig(StrictBaseModel):
    on_api_error_streak: int = 6
    on_rejected_orders_streak: int = 4
    on_daily_dd_breach: bool = True


class TradingHoursConfig(StrictBaseModel):
    enabled: bool = False
    start: str = "00:00"
    end: str = "23:59"


class SafetyConfig(StrictBaseModel):
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    trading_hours_utc: TradingHoursConfig = Field(default_factory=TradingHoursConfig)


class LoggingConfig(StrictBaseModel):
    level: str = "INFO"
    structured_json: bool = True
    save_decisions: bool = True
    save_orderbook_snapshots: bool = False
    metrics_output: str = "data/metrics.csv"
    trades_output: str = "data/trades.csv"


class ProfileConfig(StrictBaseModel):
    risk: Optional[RiskConfig] = None
    market_filters: Optional[MarketFiltersConfig] = None
    entry_rules: Optional[EntryRulesConfig] = None

class StrategyConfig(StrictBaseModel):
    environment: str = "demo"
    general: GeneralConfig
    universe: UniverseConfig
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    features: FeaturesConfig
    model: ModelConfig
    market_filters: MarketFiltersConfig
    cost_model: CostModelConfig
    entry_rules: EntryRulesConfig
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    risk: RiskConfig
    position_sizing: PositionSizingConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    profiles: Dict[str, ProfileConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_asset_maps(self):
        enabled = self.universe.enabled_assets
        _validate_per_asset_map(
            "market_filters.min_orderbook_depth_usd",
            self.market_filters.min_orderbook_depth_usd,
            enabled,
            min_value=0.0,
            max_value=10_000_000.0,
        )
        _validate_per_asset_map(
            "market_filters.max_spread_pct",
            self.market_filters.max_spread_pct,
            enabled,
            min_value=0.0,
            max_value=100.0,
        )
        _validate_per_asset_map(
            "market_filters.max_estimated_slippage_pct",
            self.market_filters.max_estimated_slippage_pct,
            enabled,
            min_value=0.0,
            max_value=100.0,
        )
        _validate_per_asset_map(
            "entry_rules.edge_threshold_pct",
            self.entry_rules.edge_threshold_pct,
            enabled,
            min_value=0.0,
            max_value=100.0,
        )
        return self

    @staticmethod
    def _validate_profile_constraints(raw_data: Dict[str, Any]) -> None:
        profiles = raw_data.get("profiles", {})
        demo = profiles.get("demo", {})
        real = profiles.get("real", {})

        demo_risk = demo.get("risk", {})
        real_risk = real.get("risk", {})

        risk_pairs = [
            ("risk_per_trade_pct", "real risk_per_trade_pct must be <= demo"),
            ("max_total_exposure_pct", "real max_total_exposure_pct must be <= demo"),
            ("max_daily_drawdown_pct", "real max_daily_drawdown_pct must be <= demo"),
        ]
        for key, msg in risk_pairs:
            if key in demo_risk and key in real_risk and real_risk[key] > demo_risk[key]:
                raise ValueError(msg)

        demo_entry = demo.get("entry_rules", {}).get("edge_threshold_pct", {})
        real_entry = real.get("entry_rules", {}).get("edge_threshold_pct", {})
        for asset, demo_value in demo_entry.items():
            if asset in real_entry and real_entry[asset] < demo_value:
                raise ValueError(
                    f"real edge threshold for {asset} must be >= demo threshold"
                )

        demo_spread = demo.get("market_filters", {}).get("max_spread_pct", {})
        real_spread = real.get("market_filters", {}).get("max_spread_pct", {})
        for asset, demo_value in demo_spread.items():
            if asset in real_spread and real_spread[asset] > demo_value:
                raise ValueError(
                    f"real max spread for {asset} must be <= demo max spread"
                )
    
    @classmethod
    def load(cls, config_path: str = "strategy_config.yaml") -> "StrategyConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML configuration {config_path} not found.")
            
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict):
            raise ValueError("YAML configuration must parse to a mapping object")
            
        environment = raw_data.get("environment", "demo")
        if environment not in {"demo", "real"}:
            raise ValueError("environment must be 'demo' or 'real'")

        cls._validate_profile_constraints(raw_data)
        
        # Pull profile overrides
        profiles = raw_data.get("profiles", {})
        active_profile = profiles.get(environment, {})
        if not active_profile:
            raise ValueError(f"missing profile overrides for environment '{environment}'")
        
        # Merge active profile overrides dynamically onto the base config structure
        if active_profile:
            raw_data = _deep_merge(raw_data, active_profile)

        # Parse using strictly typed bindings
        return cls(**raw_data)
