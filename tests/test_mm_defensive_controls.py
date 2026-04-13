from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.config import Settings
from src.core.database import Database
from src.polymarket.models import Market, MarketToken, OrderBook, PriceLevel
from src.strategy.market_maker import MarketMaker


_CID = "0x" + "9" * 64
_YES_TID = "yes_token_mm_def"


def _make_market() -> Market:
    return Market(
        condition_id=_CID,
        question_id="qid-mm-def",
        question="Will BTC stay in range this week?",
        tokens=[
            MarketToken(token_id=_YES_TID, outcome="Yes", price=0.60),
            MarketToken(token_id="no_tok", outcome="No", price=0.40),
        ],
        volume=1_000_000,
        volume_24hr=600_000,
        liquidity=700_000,
    )


def _make_book(
    *,
    best_bid: float = 0.60,
    best_ask: float = 0.62,
    depth_per_level: float = 200.0,
) -> OrderBook:
    bids = [PriceLevel(price=best_bid - i * 0.01, size=depth_per_level) for i in range(5)]
    asks = [PriceLevel(price=best_ask + i * 0.01, size=depth_per_level) for i in range(5)]
    return OrderBook(market=_CID, asset_id=_YES_TID, bids=bids, asks=asks)


@pytest.mark.asyncio
async def test_entry_price_band_blocks_outside_range(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 100.0
    settings.mm_min_entry_price = 0.25
    settings.mm_max_entry_price = 0.75

    mock_clob.get_trades.return_value = [{"price": "0.10"}] * 5
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.10, best_ask=0.12)

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: _make_market()})

    summary = await maker.run_tick()

    assert summary.get("buys_posted", 0) == 0


@pytest.mark.asyncio
async def test_mm_specific_stop_loss_is_used(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 100.0
    settings.mm_min_entry_price = 0.10
    settings.mm_max_entry_price = 0.90
    settings.stop_loss_pct = 0.30
    settings.mm_stop_loss_pct = 0.05

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: _make_market()})

    mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.60, best_ask=0.62)

    await maker.run_tick()  # IDLE -> BUY_POSTED

    mock_clob.get_order_book.return_value = _make_book(best_bid=0.59, best_ask=0.60)
    await maker.run_tick()  # BUY fill -> SELL_POSTED

    slot = maker._slots[_CID]
    assert slot.state == "SELL_POSTED"

    # With mm_stop_loss_pct=5%, stop is 0.57 for a 0.60 entry.
    # Mid at 0.56 should trigger MM stop even though generic stop_loss_pct is 30%.
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.55, best_ask=0.57)
    summary = await maker.run_tick()

    assert summary.get("losses", 0) == 1
    assert slot.state == "IDLE"


@pytest.mark.asyncio
async def test_time_stop_closes_stale_sell(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 100.0
    settings.mm_min_entry_price = 0.10
    settings.mm_max_entry_price = 0.90
    settings.mm_max_hold_seconds = 1
    settings.mm_stop_loss_pct = 0.10

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: _make_market()})

    mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.60, best_ask=0.62)

    await maker.run_tick()  # IDLE -> BUY_POSTED

    mock_clob.get_order_book.return_value = _make_book(best_bid=0.59, best_ask=0.60)
    await maker.run_tick()  # BUY fill -> SELL_POSTED

    slot = maker._slots[_CID]
    assert slot.state == "SELL_POSTED"

    # Simulate an old position that overstayed max hold time.
    slot.bought_at -= 10

    # Price is not at target sell and not at stop-loss level.
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.59, best_ask=0.61)
    summary = await maker.run_tick()

    assert summary.get("losses", 0) == 1
    assert slot.state == "IDLE"


@pytest.mark.asyncio
async def test_low_volume_or_liquidity_market_is_blocked(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 100.0
    settings.mm_min_market_volume_24h_usd = 250_000
    settings.mm_min_market_liquidity_usd = 300_000

    low_quality_market = Market(
        condition_id=_CID,
        question_id="qid-mm-low-quality",
        question="Will low-liquidity market be blocked?",
        tokens=[
            MarketToken(token_id=_YES_TID, outcome="Yes", price=0.60),
            MarketToken(token_id="no_tok", outcome="No", price=0.40),
        ],
        volume=50_000,
        volume_24hr=100_000,
        liquidity=120_000,
    )

    mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.60, best_ask=0.62)

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: low_quality_market})

    summary = await maker.run_tick()

    assert summary.get("buys_posted", 0) == 0


@pytest.mark.asyncio
async def test_performance_blacklist_blocks_consistently_losing_market(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 100.0
    settings.max_consecutive_losses = 10
    settings.mm_blacklist_enabled = True
    settings.mm_blacklist_min_trades = 3
    settings.mm_blacklist_lookback_trades = 6
    settings.mm_blacklist_min_win_rate = 0.40

    for _ in range(3):
        rid = await temp_db.insert_mm_round(
            market_id=_CID,
            token_id=_YES_TID,
            question="seed losing rounds",
            buy_price=0.60,
            shares=100.0,
        )
        await temp_db.close_mm_round(rid, sell_price=0.55, realized_pnl=-5.0)

    mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5
    mock_clob.get_order_book.return_value = _make_book(best_bid=0.60, best_ask=0.62)

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: _make_market()})

    summary = await maker.run_tick()

    assert summary.get("buys_posted", 0) == 0
