import math
from typing import Optional
from src.core.yaml_config import PositionSizingConfig, RiskConfig

class PositionSizingParams:
    def __init__(self, risk_cfg: RiskConfig, sizing_cfg: PositionSizingConfig):
        self.bankroll = risk_cfg.bankroll_usd
        self.r_base = risk_cfg.risk_per_trade_pct / 100.0
        self.vol_target = sizing_cfg.target_volatility_reference
        self.min_usd = sizing_cfg.min_notional_usd
        self.max_usd = sizing_cfg.max_notional_usd_per_trade
        self.edge_scale_cap = sizing_cfg.edge_scale_cap
        self.method = sizing_cfg.method


def calculate_dynamic_size(
    params: PositionSizingParams,
    edge_net: float,
    theta_asset: float,
    vol_asset: Optional[float] = None
) -> float:
    """
    Computes dynamically scaled position size based on YAML parameters.
    
    Parameters
    ----------
    params      : Aggregated risk and sizing constraints from configuration.
    edge_net    : Net expected value after transaction costs
    theta_asset : Minimum required edge threshold
    vol_asset   : Rolling volatility tracker for the asset (optional)
    
    Returns
    -------
    Position size in USD that respects maximum constraints
    """
    if edge_net <= 0 or theta_asset <= 0:
        return 0.0

    # Fallback to standard baseline if vol estimation fails / is disabled
    if vol_asset is None or vol_asset <= 0:
        vol_asset = params.vol_target

    # Determine scale multiplier limited by YAML cap
    edge_ratio = edge_net / theta_asset
    clamped_ratio = min(params.edge_scale_cap, edge_ratio)

    risk_allocation_usd = params.bankroll * params.r_base
    
    # Calculate initial sizing based on method structure mapped in YAML
    if params.method == "vol_adjusted_edge_scaled":
        vol_scalar = params.vol_target / vol_asset
        size = risk_allocation_usd * clamped_ratio * vol_scalar
    else:
        # standard sizing
        size = risk_allocation_usd * clamped_ratio

    # Enforce global Hard Limits
    if size < params.min_usd:
        return 0.0
    
    return min(size, params.max_usd)
