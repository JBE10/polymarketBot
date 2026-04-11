import math
from src.storage.models import Features
from src.core.yaml_config import ModelConfig

def sigmoid(x: float, temperature: float = 1.0) -> float:
    """Logistic sigmoid transformation."""
    return 1.0 / (1.0 + math.exp(-x / temperature))

def compute_p_model_up(f: Features, model_config: ModelConfig) -> float:
    """
    Computes the predicted probability of an 'UP' resolution mapping raw continuous and 
    binary features across an aggregated weight parameter.
    
    Parameters
    ----------
    f            : Features dataclass representing EMA, RSI, Momentum and ATR
    model_config : Model configuration including weights and logistic temperature 
                   from StrategyConfig.
                   
    Returns
    -------
    p_model : float between 0..1 representing predicted chance of UP
    """
    weights = model_config.weights
    
    # Simple normalized signals based on standard derivations
    ema_signal = 1.0 if f.ema_fast > f.ema_slow else -1.0
    rsi_signal = (f.rsi - 50.0) / 10.0           # Normalize ~ [-5, +5] range
    mom_signal = f.momentum                      # Assumed pre-centered by data layer
    vol_signal = (f.atr_pctile - 50.0) / 25.0    # Directional volatility indicator

    # Aggregated Linear Score
    score = (
        (weights.ema_signal * ema_signal) +
        (weights.rsi_signal * rsi_signal) +
        (weights.momentum * mom_signal) +
        (weights.vol_regime * vol_signal)
    )
    
    return sigmoid(score, temperature=model_config.logistic_temperature)
