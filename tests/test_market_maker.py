"""
A.3 — Market Maker state machine tests.

Tests the full lifecycle of MarketMaker slots:
  - IDLE → BUY_POSTED → BOUGHT → SELL_POSTED → IDLE (profit)
  - Stop-loss path
  - Regime filter blocking entry
  - Book depth < minimum blocking entry
  - Slot reconciliation (add/remove markets)

All CLOB calls are mocked — no network required.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.core.config import Settings
from src.core.database import Database
from src.polymarket.models import (
    Market,
    MarketToken,
    OrderBook,
    OrderResponse,
    OrderStatus,
    PriceLevel,
    Side,
)
from src.strategy.market_maker import MarketMaker, _MarketSlot


# ── Helpers ───────────────────────────────────────────────────────────────────

_CID = "0x" + "a" * 64
_YES_TID = "yes_token_mm"


def _make_market(cid: str = _CID) -> Market:
    return Market(
        condition_id=cid,
        question_id="qid",
        question="Will ETH exceed $5,000?",
        tokens=[
            MarketToken(token_id=_YES_TID, outcome="Yes", price=0.60),
            MarketToken(token_id="no_tok", outcome="No", price=0.40),
        ],
        volume=100_000,
        volume_24hr=10_000,
        liquidity=50_000,
    )


def _make_book(
    best_bid: float = 0.60,
    best_ask: float = 0.62,
    depth_per_level: float = 200.0,
) -> OrderBook:
    """Build an order book with 5 levels on each side."""
    bids = [PriceLevel(price=best_bid - i * 0.01, size=depth_per_level) for i in range(5)]
    asks = [PriceLevel(price=best_ask + i * 0.01, size=depth_per_level) for i in range(5)]
    return OrderBook(market=_CID, asset_id=_YES_TID, bids=bids, asks=asks)


# ── Full lifecycle: profit path ───────────────────────────────────────────────


class TestProfitLifecycle:
    """IDLE → BUY_POSTED → BOUGHT → SELL_POSTED → IDLE (profit)."""

    @pytest.mark.asyncio
    async def test_full_round_trip(self, settings: Settings, temp_db: Database, mock_clob: AsyncMock):
        """A complete profitable round: buy, fill, sell, fill."""
        settings.spread_target = 0.02
        settings.min_book_depth_usd = 100.0

        # Override mock to return our book
        book = _make_book(best_bid=0.60, best_ask=0.62, depth_per_level=200)
        mock_clob.get_order_book.return_value = book
        mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        market = _make_market()
        maker.update_active_markets({_CID: market})

        # Tick 1: IDLE → BUY_POSTED
        summary = await maker.run_tick()
        assert summary["buys_posted"] == 1
        slot = maker._slots[_CID]
        assert slot.state == "BUY_POSTED"
        assert slot.buy_price == 0.60

        # Tick 2: simulate BUY fill (ask drops to buy_price)
        fill_book = _make_book(best_bid=0.59, best_ask=0.60, depth_per_level=200)
        mock_clob.get_order_book.return_value = fill_book
        summary = await maker.run_tick()
        assert summary["sells_posted"] == 1
        assert slot.state == "SELL_POSTED"
        assert slot.sell_price == 0.62  # buy_price + spread_target

        # Tick 3: simulate SELL fill (bid rises to sell_price)
        profit_book = _make_book(best_bid=0.62, best_ask=0.64, depth_per_level=200)
        mock_clob.get_order_book.return_value = profit_book
        summary = await maker.run_tick()
        assert summary["profits"] == 1
        assert slot.state == "IDLE"

        # Verify DB has a CLOSED round
        rounds = await temp_db.get_active_mm_rounds()
        assert len(rounds) == 0  # closed, so not in active

        daily_pnl = await temp_db.get_daily_mm_pnl()
        assert daily_pnl > 0


# ── Stop-loss path ────────────────────────────────────────────────────────────


class TestStopLossPath:
    """SELL_POSTED → stop hit → IDLE (loss)."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, settings: Settings, temp_db: Database, mock_clob: AsyncMock):
        settings.spread_target = 0.02
        settings.stop_loss_pct = 0.10
        settings.min_book_depth_usd = 100.0

        book = _make_book(best_bid=0.60, best_ask=0.62, depth_per_level=200)
        mock_clob.get_order_book.return_value = book
        mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        market = _make_market()
        maker.update_active_markets({_CID: market})

        # Tick 1: IDLE → BUY_POSTED
        await maker.run_tick()

        # Tick 2: BUY fills
        fill_book = _make_book(best_bid=0.59, best_ask=0.60, depth_per_level=200)
        mock_clob.get_order_book.return_value = fill_book
        await maker.run_tick()

        slot = maker._slots[_CID]
        assert slot.state == "SELL_POSTED"

        # Tick 3: Price crashes → stop-loss
        # stop = buy_price * (1 - 0.10) = 0.60 * 0.90 = 0.54
        crash_book = _make_book(best_bid=0.53, best_ask=0.54, depth_per_level=200)
        mock_clob.get_order_book.return_value = crash_book
        summary = await maker.run_tick()
        assert summary["losses"] == 1
        assert slot.state == "IDLE"


# ── Regime filter blocking ────────────────────────────────────────────────────


class TestRegimeFiltering:
    """Regime filter should block entry when conditions are unsafe."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_entry(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        """If consecutive losses >= max, new buys should be blocked."""
        settings.max_consecutive_losses = 2
        settings.min_book_depth_usd = 100.0

        book = _make_book(best_bid=0.60, best_ask=0.62, depth_per_level=200)
        mock_clob.get_order_book.return_value = book
        mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        market = _make_market()
        maker.update_active_markets({_CID: market})

        # Seed 2 consecutive losses in DB
        for _ in range(2):
            rid = await temp_db.insert_mm_round(
                market_id=_CID, token_id=_YES_TID,
                question="test", buy_price=0.60, shares=100,
            )
            await temp_db.close_mm_round(rid, sell_price=0.55, realized_pnl=-5.0)

        # Attempt to enter — should be blocked by circuit breaker
        summary = await maker.run_tick()
        assert summary.get("buys_posted", 0) == 0


# ── Book depth blocking ──────────────────────────────────────────────────────


class TestBookDepthBlocking:
    """Thin order books should prevent entry."""

    @pytest.mark.asyncio
    async def test_thin_book_blocks(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        settings.min_book_depth_usd = 1000.0  # high threshold

        # Thin book with only $10 of depth
        thin_book = _make_book(best_bid=0.60, best_ask=0.62, depth_per_level=2.0)
        mock_clob.get_order_book.return_value = thin_book
        mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        market = _make_market()
        maker.update_active_markets({_CID: market})

        summary = await maker.run_tick()
        assert summary.get("buys_posted", 0) == 0

    @pytest.mark.asyncio
    async def test_no_book_blocks(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        """None order book should not crash — just skip."""
        mock_clob.get_order_book.return_value = None
        mock_clob.get_trades.return_value = []

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _make_market()})

        summary = await maker.run_tick()
        assert summary.get("buys_posted", 0) == 0


# ── Slot reconciliation ──────────────────────────────────────────────────────


class TestSlotReconciliation:
    """update_active_markets() should add/remove slots correctly."""

    @pytest.mark.asyncio
    async def test_add_new_market(self, settings: Settings, temp_db: Database, mock_clob: AsyncMock):
        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        assert len(maker._slots) == 0

        maker.update_active_markets({_CID: _make_market()})
        maker._reconcile_slots()
        assert _CID in maker._slots

    @pytest.mark.asyncio
    async def test_remove_unapproved_idle_slot(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _make_market()})
        maker._reconcile_slots()
        assert _CID in maker._slots

        # Now remove the market from approved set
        maker.update_active_markets({})
        maker._reconcile_slots()
        assert _CID not in maker._slots

    @pytest.mark.asyncio
    async def test_keep_active_slot_when_unapproved(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        """A slot with an active order (not IDLE) should NOT be removed."""
        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _make_market()})
        maker._reconcile_slots()

        # Force slot to non-idle state
        maker._slots[_CID].state = "BUY_POSTED"

        # Remove from approved
        maker.update_active_markets({})
        maker._reconcile_slots()

        # Should still be there because state != IDLE
        assert _CID in maker._slots

    @pytest.mark.asyncio
    async def test_max_slots_enforced(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        """Should not exceed max_mm_markets slots."""
        settings.max_mm_markets = 2

        markets = {}
        for i in range(5):
            cid = f"0x{i:064x}"
            markets[cid] = Market(
                condition_id=cid,
                question_id=f"q{i}",
                question=f"Market {i}?",
                tokens=[MarketToken(token_id=f"tok_{i}", outcome="Yes", price=0.50)],
            )

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets(markets)
        maker._reconcile_slots()

        assert len(maker._slots) <= 2


# ── Cancel all on shutdown ────────────────────────────────────────────────────


class TestCancelAll:
    """cancel_all() should reset all slots."""

    @pytest.mark.asyncio
    async def test_cancel_all_resets_slots(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        settings.min_book_depth_usd = 100.0
        book = _make_book(best_bid=0.60, best_ask=0.62, depth_per_level=200)
        mock_clob.get_order_book.return_value = book
        mock_clob.get_trades.return_value = [{"price": "0.60"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _make_market()})

        await maker.run_tick()
        assert maker._slots[_CID].state == "BUY_POSTED"

        await maker.cancel_all()
        assert maker._slots[_CID].state == "IDLE"


# ── Edge price filtering ─────────────────────────────────────────────────────


class TestEdgePrices:
    """Prices at market extremes (≤0.01 or ≥0.99) should be skipped."""

    @pytest.mark.asyncio
    async def test_extreme_low_price_skipped(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        settings.min_book_depth_usd = 0.0
        book = _make_book(best_bid=0.01, best_ask=0.03, depth_per_level=200)
        mock_clob.get_order_book.return_value = book
        mock_clob.get_trades.return_value = [{"price": "0.01"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _make_market()})

        summary = await maker.run_tick()
        assert summary.get("buys_posted", 0) == 0

    @pytest.mark.asyncio
    async def test_extreme_high_price_skipped(
        self, settings: Settings, temp_db: Database, mock_clob: AsyncMock
    ):
        settings.min_book_depth_usd = 0.0
        book = _make_book(best_bid=0.99, best_ask=1.00, depth_per_level=200)
        mock_clob.get_order_book.return_value = book
        mock_clob.get_trades.return_value = [{"price": "0.99"}] * 5

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _make_market()})

        summary = await maker.run_tick()
        assert summary.get("buys_posted", 0) == 0
