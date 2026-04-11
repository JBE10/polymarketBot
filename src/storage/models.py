from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketState:
    asset: str
    market_id: str
    expiry_utc: datetime
    p_market_up: float          # 0..1
    spread_pct: float           # %
    depth_usd: float            # side depth usable
    best_bid: float
    best_ask: float

@dataclass
class Features:
    ema_fast: float
    ema_slow: float
    rsi: float
    momentum: float
    atr_pctile: float

@dataclass
class Decision:
    asset: str
    side: str                   # "UP" or "DOWN"
    p_model_up: float
    p_market_up: float
    edge_net_pct: float
    score: float
    notional_usd: float
    reason: str
