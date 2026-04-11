from dataclasses import dataclass
from src.core.yaml_config import StrategyConfig

@dataclass
class CostConfig:
    fee: float
    slippage: float
    theta: float
    
    @property
    def total_cost(self) -> float:
        return self.fee + self.slippage


def get_cost_model(asset_symbol: str, config: StrategyConfig) -> CostConfig:
    """
    Returns the associated fee, slippage, and theta_asset dynamically extracted from the YAML strategy configuration.
    """
    symbol = asset_symbol.upper()
    cost_base = config.cost_model
    
    fee = (cost_base.taker_fee_pct / 100.0) + (cost_base.extra_buffer_pct / 100.0)
    
    filters = config.market_filters
    rules = config.entry_rules
    
    slippage_pct = filters.max_estimated_slippage_pct.get(symbol, 1.5)
    theta_pct = rules.edge_threshold_pct.get(symbol, 6.0)
    
    slippage = slippage_pct / 100.0
    theta = theta_pct / 100.0
        
    return CostConfig(fee=fee, slippage=slippage, theta=theta)


def compute_edge_net(p_model_up: float, p_market_up: float, side: str, cost: CostConfig) -> float:
    """
    Calculates edge_net = (p_model - p_market) - total_cost factoring directional parameters.
    
    Parameters
    ----------
    p_model_up  : The model's theoretical prediction probability for an UP resolution.
    p_market_up : The market binary price mapping to UP.
    side        : "UP" or "DOWN" denoting which directional share we evaluate.
    cost        : Cost structure mapping slippage and fees.
    """
    if side == "UP":
        edge_raw = (p_model_up - p_market_up)
    else:
        # Use complementary probability for the "DOWN" shares
        edge_raw = ((1.0 - p_model_up) - (1.0 - p_market_up))
        
    return edge_raw - cost.total_cost
