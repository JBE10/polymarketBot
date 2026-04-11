import pytest
from src.storage.models import Features
from src.core.yaml_config import ModelConfig
from src.model.logistic_regression import compute_p_model_up

def test_p_model_up_positive_edge():
    """
    Tests mathematical boundaries mapping logic correctly.
    """
    feat = Features(
        ema_fast=105.0,
        ema_slow=100.0,
        rsi=65.0,
        momentum=0.5,
        atr_pctile=60.0
    )
    
    config = ModelConfig()
    config.weights.ema_signal = 0.5
    config.weights.rsi_signal = 0.3
    config.weights.momentum = 0.1
    config.weights.vol_regime = 0.1
    
    p = compute_p_model_up(feat, config)
    assert p > 0.5, "Bullish features should result in > 50% probability."

def test_p_model_up_negative_edge():
    """
    Validates bearish convergence mapping output correctly.
    """
    feat = Features(
        ema_fast=95.0,
        ema_slow=100.0,
        rsi=35.0,
        momentum=-0.5,
        atr_pctile=60.0
    )
    
    config = ModelConfig()
    config.weights.ema_signal = 0.5
    config.weights.rsi_signal = 0.3
    
    p = compute_p_model_up(feat, config)
    assert p < 0.5, "Bearish features should result in < 50% probability."
