from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.database import Database
from src.core.config import Settings
from src.polymarket.models import Market, MarketToken, OrderBook, PriceLevel
from src.strategy.market_maker import MarketMaker


_CID = "0x" + "7" * 64
_YES_TID = "yes_token_mm_rej"


def _market() -> Market:
    return Market(
        condition_id=_CID,
        question_id="qid-mm-rej",
        question="Will rejection metrics be persisted?",
        tokens=[
            MarketToken(token_id=_YES_TID, outcome="Yes", price=0.55),
            MarketToken(token_id="no_token_mm_rej", outcome="No", price=0.45),
        ],
        volume=500_000,
        volume_24hr=300_000,
        liquidity=300_000,
    )


def _book(best_bid: float, best_ask: float, bid_size: float = 200.0, ask_size: float = 200.0) -> OrderBook:
    bids = [PriceLevel(price=best_bid - i * 0.01, size=bid_size) for i in range(5)]
    asks = [PriceLevel(price=best_ask + i * 0.01, size=ask_size) for i in range(5)]
    return OrderBook(market=_CID, asset_id=_YES_TID, bids=bids, asks=asks)


@pytest.mark.asyncio
async def test_rejection_metrics_records_price_band(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 50.0
    settings.mm_min_entry_price = 0.30
    settings.mm_max_entry_price = 0.80

    mock_clob.get_trades.return_value = [{"price": "0.10"}] * 5
    mock_clob.get_order_book.return_value = _book(best_bid=0.10, best_ask=0.12)

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: _market()})

    await maker.run_tick()

    rejections = await temp_db.get_mm_rejections(since_hours=24, limit=20)
    assert any(r.get("reason_code") == "price_band" for r in rejections)


@pytest.mark.asyncio
async def test_rejection_metrics_records_micro_obi(
    settings: Settings,
    temp_db: Database,
    mock_clob: AsyncMock,
) -> None:
    settings.min_book_depth_usd = 50.0
    settings.mm_min_entry_price = 0.10
    settings.mm_max_entry_price = 0.90

    # Create strong sell-pressure OBI by making ask size much larger than bid size.
    mock_clob.get_trades.return_value = [{"price": "0.55"}] * 5
    mock_clob.get_order_book.return_value = _book(
        best_bid=0.55,
        best_ask=0.59,
        bid_size=100.0,
        ask_size=2000.0,
    )

    maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
    maker.update_active_markets({_CID: _market()})

    await maker.run_tick()

    rejections = await temp_db.get_mm_rejections(since_hours=24, limit=20)
    assert any(r.get("reason_code") == "obi" for r in rejections)
