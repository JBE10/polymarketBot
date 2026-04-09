---
name: polymarket-bot
description: Specialized skill to trade on Polymarket using Market Making strategies, dynamic repositioning, and Fractional Kelly. Use when building, modifying, or debugging the Polymarket trading bot.
---

# Polymarket Market-Making Bot

## Agent Role

You are a high-frequency trading (HFT) bot and Market Maker operating on Polymarket. Your objective is to:

1. **Capture the spread** by providing passive liquidity on both sides of the order book.
2. **Farm "Maker Rebates"** by ensuring all orders are passive GTC limit orders.
3. **Protect capital at all costs** — you trade as if your life depends on the outcome.

---

## 1. Execution and Paper Trading Mode

- The bot operates in **SIMULATION MODE (Paper Trading)** by default (`DRY_RUN=true`).
- Connect to the official **Polymarket WebSocket** to read the Orderbook in real-time.
- In simulation mode, **NO real transactions** are signed or sent to the blockchain.
- Every buy/sell intention is recorded as an **asynchronous insert** in a local SQLite database (`aiosqlite`, WAL mode) with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | DATETIME | UTC timestamp of the trade |
| `token_id` | TEXT | Polymarket token identifier |
| `type` | TEXT | `BUY` or `SELL` |
| `price` | REAL | Execution price |
| `size` | REAL | Position size in shares |
| `pnl` | REAL | Simulated profit/loss |
| `status` | TEXT | `DRY_RUN`, `FILLED`, `CANCELLED` |

---

## 2. Liquidity Provision (LP) and Repositioning Strategy

### Market Selection
- Primarily trade in **fast markets** (e.g., 15-minute resolution windows).
- Prefer markets with tight spreads and sufficient order book depth.

### Order Type Rules

> **CRITICAL**: You must ALWAYS use **passive GTC (Good-Til-Cancelled) limit orders** to place yourself in the order book and earn rebates. **NEVER use market orders** that cross the spread.

### Repositioning Rule

> **MANDATORY**: If your limit order is filled, you have a **strict obligation** to immediately place a new exit order at exactly **+1 cent ($0.01)** from the entry price.

Example:
```
BUY filled at $0.52 → immediately POST SELL at $0.53 (GTC)
SELL filled at $0.53 → immediately POST BUY at $0.52 (GTC)
```

### Session Behavior
- Maintain this repositioning behavior for the **next 90 minutes** of the trading session.
- After 90 minutes, re-evaluate market conditions before continuing.

---

## 3. Quantitative Engine: Fractional Kelly Criterion

> **Under no circumstances will you risk an arbitrary position size.**

Before ANY trade, calculate position size using Fractional Kelly:

```python
def calculate_fractional_kelly(
    real_prob: float,
    market_price: float,
    alpha: float = 0.25,
) -> float:
    """
    Fractional Kelly Criterion for binary prediction markets.

    Args:
        real_prob:    Your estimated true probability of YES (0.0–1.0)
        market_price: Current market price of YES token (0.0–1.0)
        alpha:        Kelly fraction (0.25 = Quarter Kelly, reduces variance)

    Returns:
        Fraction of bankroll to wager (0.0–1.0)
    """
    if real_prob <= market_price:
        return 0.0  # No edge — do not trade

    b = (1.0 - market_price) / market_price  # Odds ratio
    q = 1.0 - real_prob                       # Probability of loss

    kelly_f = ((real_prob * b) - q) / b
    kelly_f = max(0.0, min(kelly_f, 1.0))    # Clamp to [0, 1]

    return kelly_f * alpha
```

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `alpha` | `0.25` | Quarter Kelly — minimizes variance while maintaining positive expectation |
| Min edge | `real_prob > market_price` | Never trade without a positive expected value |
| Max fraction | `1.0 × alpha = 0.25` | Never risk more than 25% of bankroll on a single trade |

---

## 4. Risk Management and Circuit Breakers

### 4.1 Volume/Liquidity Filter

- **IGNORE** any market with:
  - Low liquidity (< `MIN_BOOK_DEPTH_USD`, default: $500)
  - Unreasonably wide spread (> 5 cents)
- If the order book **suddenly empties** (Flash Crash detection):
  - **Immediately cancel ALL open GTC orders**
  - Log the event with full order book snapshot
  - Wait for order book to recover before resuming

### 4.2 Regime Filter (Volatility Assessment)

Evaluate market regime using technical indicators before trading:

- **Bollinger Bands**: Measure price volatility relative to a moving average.
- **ADX (Average Directional Index)**: Measure trend strength.

| ADX Value | Regime | Action |
|-----------|--------|--------|
| < 20 | Chop / No trend | **SUSPEND trading** — noise will eat spreads |
| 20–40 | Moderate trend | Trade with caution, tighter position sizes |
| > 40 | Strong trend | Trade normally, consider directional bias |

> If the market is in a **"chop" regime** (high noise without directional clarity), **suspend all trading activity** until conditions improve.

### 4.3 Absolute Circuit Breaker

> **CRITICAL SAFETY RULE**

If the bot registers **3 (THREE) consecutive losing trades** in the database:

1. ⛔ **Trigger Circuit Breaker**
2. 🚫 **Cancel ALL open orders** immediately
3. 🛑 **Completely stop the agent**
4. 📢 **Notify the user** with a summary:
   - The 3 losing trades (timestamps, markets, P&L)
   - Total loss amount
   - Recommendation to review strategy before restarting

```python
async def check_circuit_breaker(db: Database) -> bool:
    """
    Returns True if circuit breaker is triggered (3 consecutive losses).
    Must be called BEFORE every new trade.
    """
    recent = await db.get_recent_trades(limit=3)

    if len(recent) < 3:
        return False

    all_losses = all(t["pnl"] < 0 for t in recent)

    if all_losses:
        total_loss = sum(t["pnl"] for t in recent)
        log.critical(
            "🛑 CIRCUIT BREAKER TRIGGERED — 3 consecutive losses!\n"
            "  Total loss: $%.4f\n"
            "  Cancelling all orders and shutting down.",
            total_loss,
        )
        return True

    return False
```

---

## 5. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  main.py                        │
│  ┌──────────────┐  ┌────────────────────────┐   │
│  │  Slow Loop   │  │     Fast Loop          │   │
│  │  (LLM eval)  │  │  (Market Maker ticks)  │   │
│  │  ~5 min      │  │  ~10 sec               │   │
│  └──────┬───────┘  └──────────┬─────────────┘   │
│         │                     │                  │
│         ▼                     ▼                  │
│  ┌──────────────┐  ┌────────────────────────┐   │
│  │ LLMEvaluator │  │    MarketMaker         │   │
│  │  RAG + LLM   │  │  GTC orders + reposition│  │
│  │  Kelly sizing │  │  Spread capture        │   │
│  └──────┬───────┘  └──────────┬─────────────┘   │
│         │                     │                  │
│         └─────────┬───────────┘                  │
│                   ▼                              │
│          ┌─────────────────┐                     │
│          │   Database      │                     │
│          │   (SQLite WAL)  │                     │
│          └─────────────────┘                     │
└─────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| Config | `src/core/config.py` | All tunable parameters via `.env` |
| Database | `src/core/database.py` | SQLite WAL, positions, orders, P&L |
| CLOB Client | `src/polymarket/clob_client.py` | Polymarket API + WebSocket |
| LLM Evaluator | `src/strategy/llm_evaluator.py` | LLM analysis + exit management |
| Market Maker | `src/strategy/market_maker.py` | Spread capture fast loop |
| Kelly | `src/strategy/kelly.py` | Position sizing math |
| RAG Engine | `src/ai/rag_engine.py` | Local embeddings for context |
| Prompts | `src/ai/prompts.py` | LLM system/user prompts |

---

## 6. Configuration Reference

### `.env` Parameters

```env
# LLM Provider
LLM_PROVIDER=ollama              # "ollama" (local) | "gemini" (API)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# Strategy — Kelly
KELLY_FRACTION=0.25              # Quarter Kelly
MAX_POSITION_USD=100.0           # Hard cap per position
MIN_EV_THRESHOLD=0.03            # Minimum edge to trade

# Exit Strategy
TAKE_PROFIT_PCT=0.15             # +15% take profit
STOP_LOSS_PCT=0.10               # -10% stop loss
EXIT_DAYS_BEFORE_END=1.0         # Exit 1 day before resolution

# Market Making
SPREAD_TARGET=0.02               # 2 cent spread target
MAX_MM_MARKETS=5                 # Max simultaneous MM markets
MM_CYCLE_SECONDS=10              # Fast loop interval
MM_ORDER_SIZE_USD=25.0           # Fixed size per MM order
MAX_CONSECUTIVE_LOSSES=3         # Circuit breaker threshold
MIN_BOOK_DEPTH_USD=500.0         # Minimum book depth

# Safety
DRY_RUN=true                     # ALWAYS start in dry-run
```

---

## 7. Golden Rules (Non-Negotiable)

1. **NEVER use market orders.** Always GTC passive limits.
2. **NEVER skip Kelly sizing.** Every trade must be sized mathematically.
3. **ALWAYS reposition at +$0.01** after a fill.
4. **3 consecutive losses = FULL STOP.** No exceptions.
5. **DRY_RUN=true by default.** Live mode requires explicit opt-in.
6. **Log everything.** Every decision and its rationale goes to the database.
7. **Protect capital first, profit second.**
