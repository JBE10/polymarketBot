# Strategy Spec - Multi-Asset 15m Binary Markets (API-First)

## 1. Scope
This strategy trades 15-minute directional binary markets (UP/DOWN) for:
- BTC, ETH, SOL, BNB, XRP

Execution is API-only (no web scraping).

---

## 2. Data Inputs

### 2.1 Market Venue Data (Polymarket/CLOB)
- Active 15m markets per asset
- Market expiry timestamp (UTC)
- Orderbook: best bid/ask, depth, spread
- Last traded prices and fills (if available)

### 2.2 External Market Data (Spot Exchange API)
- OHLCV 1m and 5m for each asset
- Derived returns and volatility

### 2.3 Time Constraints
- Decision cycle every 30s
- No-trade window: last 3 minutes before market expiry

---

## 3. Feature Engineering

For each asset at each decision cycle:
- EMA fast (9) on 5m close
- EMA slow (21) on 5m close
- RSI(14) on 5m close
- Momentum (last 3 bars, 5m)
- ATR percentile (volatility regime proxy)

Normalize features to stable ranges before model scoring.

---

## 4. Probability Model

### 4.1 Score
score = w1*ema_signal + w2*rsi_signal + w3*momentum + w4*vol_regime + b

Default weights:
- w1=0.35, w2=0.20, w3=0.30, w4=0.15

### 4.2 Probability Mapping
p_model_up = sigmoid(score) = 1/(1+exp(-score))

p_model_down = 1 - p_model_up

### 4.3 Calibration (required before full real scale)
Calibrate p_model with:
- Platt scaling or isotonic regression
- Out-of-sample only

---

## 5. Market-Implied Probability

For the UP contract:
- p_market_up = contract implied probability from tradable price

For DOWN:
- p_market_down = 1 - p_market_up

---

## 6. Cost Model

total_cost = fee + slippage_est + latency_buffer

Per-asset slippage priors:
- BTC: 0.7%
- ETH: 0.8%
- SOL: 1.2%
- BNB: 1.2%
- XRP: 1.5%

Latency buffer default: 0.15%

---

## 7. Entry Logic

For side in {UP, DOWN}:
- edge_raw = p_model_side - p_market_side
- edge_net = edge_raw - total_cost

Enter only if:
1) edge_net > threshold(asset, regime)
2) spread <= max_spread(asset)
3) depth >= min_depth(asset)
4) not in no-trade window
5) risk guard allows new exposure

Initial thresholds:
- BTC: 3.2%
- ETH: 3.5%
- SOL: 4.8%
- BNB: 4.8%
- XRP: 5.5%

---

## 8. Asset Selection (Multi-Asset Ranker)

Compute score per candidate:
rank_score = edge_net * liquidity_score * trend_quality * (1 - spread_penalty)

Trade top N assets per cycle:
- N = 2 (default)

---

## 9. Position Sizing

risk_usd = bankroll * risk_per_trade_pct

size_usd = risk_usd * clamp(edge_net / threshold, 0, 2.0) * (target_vol / asset_vol)

Apply hard caps:
- max_notional_per_trade
- max_exposure_per_asset
- max_total_exposure

---

## 10. Risk Management

Hard limits:
- risk_per_trade: 0.5% (demo), 0.3% (real initial)
- max_total_exposure: 2.5% (demo), 1.8% (real)
- max_daily_drawdown: 3.0% (demo), 2.2% (real)

Behavioral guards:
- cooldown after 2 consecutive losses in same asset
- kill-switch on repeated API/order failures

---

## 11. Execution Policy

- Prefer limit orders
- Cancel/requote if not filled within TIF window
- Max requotes: 2
- Idempotent order submission keys required

No market order during high spread/depth deterioration.

---

## 12. Observability & Logging

Log every decision:
- features, p_model, p_market, edge_net, costs
- filters pass/fail reasons
- order attempts/fills/rejects
- realized slippage vs estimated slippage

Daily reports by:
- asset
- regime
- hour-of-day
- side (UP/DOWN)

---

## 13. Validation Gates

Promotion Demo -> Real requires:
- >= 150 trades in paper mode
- positive EV/trade
- Profit Factor > 1.15
- max DD within configured limits
- no severe operational incidents (API/order reliability)

---

## 14. Change Control

Any parameter change must include:
- rationale
- expected impact
- backtest + paper evidence
- version bump in config and changelog
