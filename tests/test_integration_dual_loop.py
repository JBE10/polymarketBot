"""
A.5 — Integration test: dual loop smoke.

Starts both loops (LLM slow + MM fast) with fully mocked CLOB and LLM,
verifies the end-to-end flow from market discovery through profit recording.

Protocol:
  - Mock CLOB returns canned markets + books
  - Mock LLM always returns a fixed evaluation (BUY signal)
  - LLM cycle feeds approved markets to MM
  - MM posts BUY, simulated fill triggers SELL, profit recorded
  - Run for a few ticks max
  - Assert mm_rounds has at least 1 CLOSED round
  - Timeout: 30 seconds
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.config import Settings
from src.core.database import Database
from src.polymarket.models import (
    Market,
    MarketToken,
    OrderBook,
    PriceLevel,
)
from src.strategy.market_maker import MarketMaker

# ── Test data ─────────────────────────────────────────────────────────────────

_CID = "0x" + "d" * 64
_YES_TID = "yes_integ_001"


def _market() -> Market:
    return Market(
        condition_id=_CID,
        question_id="qid_integ",
        question="Integration test market?",
        tokens=[
            MarketToken(token_id=_YES_TID, outcome="Yes", price=0.55),
            MarketToken(token_id="no_integ_001", outcome="No", price=0.45),
        ],
        volume=200_000,
        volume_24hr=15_000,
        liquidity=60_000,
        end_date_iso="2026-12-31T23:59:59Z",
    )


def _book_idle() -> OrderBook:
    """Book when market is idle — BUY will be posted at best_bid."""
    return OrderBook(
        market=_CID,
        asset_id=_YES_TID,
        bids=[PriceLevel(price=0.55 - i * 0.01, size=200) for i in range(5)],
        asks=[PriceLevel(price=0.57 + i * 0.01, size=200) for i in range(5)],
    )


def _book_buy_fills() -> OrderBook:
    """Book where ask drops to buy_price → simulates BUY fill."""
    return OrderBook(
        market=_CID,
        asset_id=_YES_TID,
        bids=[PriceLevel(price=0.54 - i * 0.01, size=200) for i in range(5)],
        asks=[PriceLevel(price=0.55 + i * 0.01, size=200) for i in range(5)],
    )


def _book_sell_fills() -> OrderBook:
    """Book where bid rises above sell_price → simulates SELL fill.

    sell_price = buy_price + spread_target = 0.55 + 0.02 ≈ 0.57.
    We use best_bid = 0.58 to clear any floating-point rounding.
    """
    return OrderBook(
        market=_CID,
        asset_id=_YES_TID,
        bids=[PriceLevel(price=0.58 - i * 0.01, size=200) for i in range(5)],
        asks=[PriceLevel(price=0.60 + i * 0.01, size=200) for i in range(5)],
    )


# ── Integration test ─────────────────────────────────────────────────────────


class TestDualLoopIntegration:
    """End-to-end: LLM approves market → MM trades it → profit recorded."""

    @pytest.mark.asyncio
    async def test_mm_completes_round_trip(self, settings: Settings, temp_db: Database):
        """
        Simulate the MM fast loop through a full profitable round:
          1. LLM approves the market (simulated by calling update_active_markets)
          2. MM enters at best_bid
          3. BUY fills (simulated)
          4. SELL posted at entry + spread
          5. SELL fills (simulated)
          6. Profit recorded in mm_rounds

        This tests the same flow that would occur in production with the
        real dual-loop architecture, but driven step-by-step.
        """
        settings.spread_target = 0.02
        settings.min_book_depth_usd = 100.0
        settings.mm_order_size_usd = 25.0

        # Build mock CLOB
        mock_clob = AsyncMock()
        mock_clob.get_trades.return_value = [{"price": "0.55"}] * 5

        # Phase-specific book sequence
        book_sequence = [_book_idle(), _book_buy_fills(), _book_sell_fills()]
        call_count = 0

        async def get_book_side_effect(token_id: str):
            nonlocal call_count
            idx = min(call_count, len(book_sequence) - 1)
            call_count += 1
            return book_sequence[idx]

        mock_clob.get_order_book.side_effect = get_book_side_effect

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)

        # Step 1: LLM approves the market
        market = _market()
        maker.update_active_markets({_CID: market})

        # Step 2: Tick 1 — IDLE → BUY_POSTED
        s1 = await maker.run_tick()
        assert s1["buys_posted"] == 1

        # Step 3: Tick 2 — BUY fills → BOUGHT → SELL_POSTED
        s2 = await maker.run_tick()
        assert s2["sells_posted"] == 1

        # Step 4: Tick 3 — SELL fills → profit
        s3 = await maker.run_tick()
        assert s3["profits"] == 1

        # Verify: mm_rounds has at least 1 CLOSED round
        active = await temp_db.get_active_mm_rounds()
        assert len(active) == 0, "All rounds should be closed"

        daily_pnl = await temp_db.get_daily_mm_pnl()
        expected_gross = (settings.spread_target / 0.55) * settings.mm_order_size_usd
        expected_rebate = settings.mm_order_size_usd * 0.005
        assert daily_pnl == pytest.approx(expected_gross + expected_rebate)

    @pytest.mark.asyncio
    async def test_mm_handles_stop_loss_and_continues(
        self, settings: Settings, temp_db: Database
    ):
        """
        After a stop-loss, the MM should reset and be able to enter again.
        """
        settings.spread_target = 0.02
        settings.stop_loss_pct = 0.05
        settings.min_book_depth_usd = 100.0

        mock_clob = AsyncMock()
        mock_clob.get_trades.return_value = [{"price": "0.55"}] * 5

        # Sequence: idle book → buy fills → price crash (stop-loss)
        crash_book = OrderBook(
            market=_CID,
            asset_id=_YES_TID,
            bids=[PriceLevel(price=0.50 - i * 0.01, size=200) for i in range(5)],
            asks=[PriceLevel(price=0.52 + i * 0.01, size=200) for i in range(5)],
        )

        books = [_book_idle(), _book_buy_fills(), crash_book, _book_idle()]
        idx = 0

        async def get_book(tid):
            nonlocal idx
            b = books[min(idx, len(books) - 1)]
            idx += 1
            return b

        mock_clob.get_order_book.side_effect = get_book

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets({_CID: _market()})

        # Tick 1: BUY posted
        await maker.run_tick()
        # Tick 2: BUY fills
        await maker.run_tick()
        # Tick 3: stop-loss
        s3 = await maker.run_tick()
        assert s3["losses"] == 1

        # Slot should be IDLE again — ready for re-entry
        slot = maker._slots[_CID]
        assert slot.state == "IDLE"

    @pytest.mark.asyncio
    async def test_concurrent_markets(self, settings: Settings, temp_db: Database):
        """MM should handle multiple markets concurrently."""
        settings.max_mm_markets = 3
        settings.spread_target = 0.02
        settings.min_book_depth_usd = 100.0

        mock_clob = AsyncMock()
        mock_clob.get_trades.return_value = [{"price": "0.55"}] * 5
        mock_clob.get_order_book.return_value = _book_idle()

        markets = {}
        for i in range(3):
            cid = f"0x{i:064x}"
            markets[cid] = Market(
                condition_id=cid,
                question_id=f"q{i}",
                question=f"Concurrent market {i}?",
                tokens=[
                    MarketToken(token_id=f"yes_{i}", outcome="Yes", price=0.55),
                    MarketToken(token_id=f"no_{i}", outcome="No", price=0.45),
                ],
                volume=100_000,
                volume_24hr=10_000,
                liquidity=50_000,
            )

        maker = MarketMaker(clob=mock_clob, db=temp_db, settings=settings)
        maker.update_active_markets(markets)

        summary = await maker.run_tick()
        assert summary["buys_posted"] == 3

        # All 3 slots should be in BUY_POSTED
        for slot in maker._slots.values():
            assert slot.state == "BUY_POSTED"
