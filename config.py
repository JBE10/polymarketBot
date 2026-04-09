from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "polymarket_intelligence.db"

# ── API Base URLs ────────────────────────────────────────────────────────────
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API  = "https://clob.polymarket.com"
POLYMARKET_DATA_API  = "https://data-api.polymarket.com"
KALSHI_API_BASE      = "https://trading-api.kalshi.com/trade-api/v2"

# ── Analytics thresholds ─────────────────────────────────────────────────────
WHALE_THRESHOLD_USD  = 5_000    # trades >= $5k are flagged as whale activity
ARB_MIN_SPREAD_PCT   = 2.0      # minimum spread % to surface arbitrage signals
SIMILARITY_THRESHOLD = 0.45     # fuzzy-match threshold for cross-platform pairing

# ── Refresh cadence (seconds) ────────────────────────────────────────────────
REFRESH_MARKETS      = 60
REFRESH_WHALES       = 30
REFRESH_ARBITRAGE    = 300
REFRESH_ORDERBOOK    = 15
REFRESH_RISK         = 300
REFRESH_PORTFOLIO    = 60
REFRESH_ALERTS       = 30

# ── HTTP ─────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT      = 12.0
MAX_MARKETS_FETCH    = 100

# ── App identity ─────────────────────────────────────────────────────────────
APP_TITLE   = "Polymarket Intelligence"
APP_VERSION = "1.0.0"
