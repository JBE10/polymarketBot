"""
Shared pytest fixtures for the Polymarket bot test suite.

Provides:
  - settings: Settings override with temp DB and DRY_RUN=true
  - temp_db: ephemeral Database connected to a temp SQLite file
  - mock_clob: AsyncMock of AsyncClobClient with canned responses
  - sample_market: reusable Market fixture with YES/NO tokens
  - sample_order_book: canned OrderBook with bids and asks
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Settings
from src.core.database import Database
from src.polymarket.models import (
    Market,
    MarketToken,
    OrderBook,
    OrderResponse,
    OrderStatus,
    PriceLevel,
)

# ── Settings override ────────────────────────────────────────────────────────


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with temp DB, dry-run enabled, and no real API keys."""
    return Settings(
        dry_run=True,
        data_dir=tmp_path / "data",
        gemini_api_key="",
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma3:4b",
        polymarket_wallet_address="0x" + "0" * 40,
        cycle_interval_seconds=5,
        mm_cycle_seconds=1,
        spread_target=0.02,
        max_mm_markets=3,
        mm_order_size_usd=25.0,
        max_consecutive_losses=3,
        min_book_depth_usd=500.0,
        kelly_fraction=0.25,
        max_position_usd=100.0,
        min_ev_threshold=0.03,
        min_confidence="MEDIUM",
        take_profit_pct=0.15,
        stop_loss_pct=0.10,
    )


# ── Temporary database ───────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def temp_db(tmp_path: Path) -> Database:
    """Ephemeral Database instance — auto-cleaned after test."""
    db_path = tmp_path / "test_bot.db"
    db = Database(db_path)
    await db.connect()
    yield db
    await db.close()


# ── Sample market data ───────────────────────────────────────────────────────

_CONDITION_ID = "0x" + "a" * 64
_YES_TOKEN_ID = "yes_token_12345"
_NO_TOKEN_ID = "no_token_67890"


@pytest.fixture
def sample_market() -> Market:
    """A realistic binary market with YES/NO tokens."""
    return Market(
        condition_id=_CONDITION_ID,
        question_id="qid_test",
        question="Will Bitcoin exceed $100,000 by end of 2026?",
        description="Resolves YES if BTC/USD closes above $100,000 on any major exchange.",
        market_slug="btc-100k-2026",
        end_date_iso="2026-12-31T23:59:59Z",
        active=True,
        closed=False,
        tokens=[
            MarketToken(token_id=_YES_TOKEN_ID, outcome="Yes", price=0.65),
            MarketToken(token_id=_NO_TOKEN_ID, outcome="No", price=0.35),
        ],
        volume=500_000.0,
        volume_24hr=25_000.0,
        liquidity=80_000.0,
    )


@pytest.fixture
def sample_order_book() -> OrderBook:
    """Canned order book with realistic bids and asks."""
    return OrderBook(
        market=_CONDITION_ID,
        asset_id=_YES_TOKEN_ID,
        bids=[
            PriceLevel(price=0.64, size=500),
            PriceLevel(price=0.63, size=800),
            PriceLevel(price=0.62, size=1200),
            PriceLevel(price=0.61, size=600),
            PriceLevel(price=0.60, size=400),
        ],
        asks=[
            PriceLevel(price=0.66, size=450),
            PriceLevel(price=0.67, size=700),
            PriceLevel(price=0.68, size=900),
            PriceLevel(price=0.69, size=500),
            PriceLevel(price=0.70, size=300),
        ],
    )


# ── Mock CLOB client ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_clob(sample_order_book: OrderBook, sample_market: Market) -> AsyncMock:
    """
    AsyncMock of AsyncClobClient with canned responses.

    - get_markets() returns [sample_market]
    - get_order_book() returns sample_order_book
    - get_trades() returns a list of canned trade dicts
    - place_order() returns a successful OrderResponse
    - get_open_orders() returns []
    - cancel_order() returns True
    """
    mock = AsyncMock()

    mock.get_markets.return_value = [sample_market]
    mock.get_market.return_value = sample_market
    mock.get_order_book.return_value = sample_order_book

    mock.get_trades.return_value = [
        {"price": "0.64", "size": "100"},
        {"price": "0.65", "size": "200"},
        {"price": "0.63", "size": "150"},
    ]

    mock.get_price.return_value = 0.65

    mock.place_order.return_value = OrderResponse(
        order_id="test_order_001",
        status=OrderStatus.MATCHED,
        transaction_hash="0x" + "b" * 64,
        filled_size=100.0,
    )

    mock.get_open_orders.return_value = []
    mock.cancel_order.return_value = True
    mock.cancel_all = AsyncMock()
    mock.close = AsyncMock()
    mock.initialize = AsyncMock()

    return mock
