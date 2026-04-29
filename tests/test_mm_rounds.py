"""
A.4 — Database mm_rounds CRUD tests.

Validates all market-making round operations:
  - insert_mm_round
  - update_mm_round_bought
  - update_mm_round_sell_posted
  - close_mm_round
  - cancel_mm_round
  - get_consecutive_losses (streak counting)
  - get_daily_mm_pnl (today-only sum)
  - get_active_mm_rounds (excludes CLOSED/CANCELLED)
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.database import Database

_MKT = "mkt_mm_test"
_TOK = "tok_mm_test"
_Q = "Will ETH hit $10k?"


class TestInsertMMRound:
    """insert_mm_round() creates a new round in BUY_POSTED state."""

    @pytest.mark.asyncio
    async def test_insert_returns_id(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT,
            token_id=_TOK,
            question=_Q,
            buy_price=0.60,
            shares=100.0,
            buy_order_id="ord_buy_001",
        )
        assert isinstance(rid, int)
        assert rid > 0

    @pytest.mark.asyncio
    async def test_initial_status_is_buy_posted(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.50, shares=50.0,
        )
        rounds = await temp_db.get_active_mm_rounds()
        assert len(rounds) == 1
        assert rounds[0]["status"] == "BUY_POSTED"
        assert rounds[0]["id"] == rid

    @pytest.mark.asyncio
    async def test_insert_without_order_id(self, temp_db: Database):
        """buy_order_id is optional (dry-run mode)."""
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.55, shares=75.0,
        )
        assert rid > 0


class TestUpdateMMRoundBought:
    """update_mm_round_bought() transitions BUY_POSTED → BOUGHT."""

    @pytest.mark.asyncio
    async def test_status_changes_to_bought(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.update_mm_round_bought(rid)

        rounds = await temp_db.get_active_mm_rounds()
        bought = [r for r in rounds if r["id"] == rid]
        assert len(bought) == 1
        assert bought[0]["status"] == "BOUGHT"


class TestUpdateSellPosted:
    """update_mm_round_sell_posted() sets sell_price and sell_order_id."""

    @pytest.mark.asyncio
    async def test_sell_posted(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.update_mm_round_bought(rid)
        await temp_db.update_mm_round_sell_posted(rid, sell_price=0.62, sell_order_id="ord_sell_001")

        rounds = await temp_db.get_active_mm_rounds()
        sell = [r for r in rounds if r["id"] == rid]
        assert len(sell) == 1
        assert sell[0]["status"] == "SELL_POSTED"
        assert sell[0]["sell_price"] == 0.62
        assert sell[0]["sell_order_id"] == "ord_sell_001"


class TestCloseMMRound:
    """close_mm_round() marks as CLOSED with P&L and removes from active."""

    @pytest.mark.asyncio
    async def test_close_with_profit(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.close_mm_round(rid, sell_price=0.62, realized_pnl=2.0, rebate_est=0.30)

        # Should NOT appear in active rounds
        active = await temp_db.get_active_mm_rounds()
        assert not any(r["id"] == rid for r in active)
        assert temp_db._db is not None
        async with temp_db._db.execute(
            "SELECT realized_pnl, rebate_est, net_pnl FROM mm_rounds WHERE id = ?",
            (rid,),
        ) as cur:
            row = await cur.fetchone()
        assert row["realized_pnl"] == 2.0
        assert row["rebate_est"] == 0.30
        assert row["net_pnl"] == 2.30

    @pytest.mark.asyncio
    async def test_close_with_loss(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.close_mm_round(rid, sell_price=0.55, realized_pnl=-5.0)

        active = await temp_db.get_active_mm_rounds()
        assert not any(r["id"] == rid for r in active)


class TestCancelMMRound:
    """cancel_mm_round() marks as CANCELLED and removes from active."""

    @pytest.mark.asyncio
    async def test_cancelled_not_in_active(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.cancel_mm_round(rid)

        active = await temp_db.get_active_mm_rounds()
        assert not any(r["id"] == rid for r in active)


class TestGetConsecutiveLosses:
    """get_consecutive_losses() counts the unbroken trailing loss streak."""

    @pytest.mark.asyncio
    async def test_streak_of_3_losses(self, temp_db: Database):
        """3 consecutive losses should return streak=3."""
        for pnl in [-1.0, -2.0, -3.0]:
            rid = await temp_db.insert_mm_round(
                market_id=_MKT, token_id=_TOK, question=_Q,
                buy_price=0.60, shares=100.0,
            )
            await temp_db.close_mm_round(rid, sell_price=0.58, realized_pnl=pnl)

        streak = await temp_db.get_consecutive_losses(_MKT)
        assert streak == 3

    @pytest.mark.asyncio
    async def test_only_losses_gives_full_streak(self, temp_db: Database):
        """All losses → streak equals the number of rounds."""
        for _ in range(4):
            rid = await temp_db.insert_mm_round(
                market_id=_MKT, token_id=_TOK, question=_Q,
                buy_price=0.60, shares=100.0,
            )
            await temp_db.close_mm_round(rid, sell_price=0.58, realized_pnl=-1.0)

        streak = await temp_db.get_consecutive_losses(_MKT)
        assert streak == 4

    @pytest.mark.asyncio
    async def test_win_then_loss_gives_streak_of_one(self, temp_db: Database):
        """Win followed by a single loss → streak = 1 (most recent is loss).

        Uses a deliberate 1s delay between the win and loss to ensure
        closed_at timestamps differ and ORDER BY is deterministic.
        """
        # Round 1: win
        r1 = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.close_mm_round(r1, sell_price=0.62, realized_pnl=+2.0)

        # Wait 1 second so closed_at is strictly later
        await asyncio.sleep(1.1)

        # Round 2: loss (most recent)
        r2 = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.close_mm_round(r2, sell_price=0.55, realized_pnl=-5.0)

        streak = await temp_db.get_consecutive_losses(_MKT)
        assert streak == 1

    @pytest.mark.asyncio
    async def test_no_rounds_returns_zero(self, temp_db: Database):
        streak = await temp_db.get_consecutive_losses("nonexistent_market")
        assert streak == 0

    @pytest.mark.asyncio
    async def test_all_wins_returns_zero(self, temp_db: Database):
        for _ in range(3):
            rid = await temp_db.insert_mm_round(
                market_id=_MKT, token_id=_TOK, question=_Q,
                buy_price=0.60, shares=100.0,
            )
            await temp_db.close_mm_round(rid, sell_price=0.62, realized_pnl=2.0)

        streak = await temp_db.get_consecutive_losses(_MKT)
        assert streak == 0

    @pytest.mark.asyncio
    async def test_per_market_isolation(self, temp_db: Database):
        """Losses in one market should not affect another market's streak."""
        mkt_a = "mkt_a"
        mkt_b = "mkt_b"

        # 2 losses in mkt_a
        for _ in range(2):
            rid = await temp_db.insert_mm_round(
                market_id=mkt_a, token_id=_TOK, question=_Q,
                buy_price=0.60, shares=100.0,
            )
            await temp_db.close_mm_round(rid, sell_price=0.58, realized_pnl=-2.0)

        # mkt_b should have 0 losses
        assert await temp_db.get_consecutive_losses(mkt_b) == 0
        assert await temp_db.get_consecutive_losses(mkt_a) == 2


class TestGetDailyMMPnl:
    """get_daily_mm_pnl() sums only today's closed rounds."""

    @pytest.mark.asyncio
    async def test_sums_today_closed(self, temp_db: Database):
        """Multiple closed rounds today should sum their net P&L."""
        for pnl, rebate in [(1.0, 0.10), (2.0, 0.20), (-0.5, 0.05)]:
            rid = await temp_db.insert_mm_round(
                market_id=_MKT, token_id=_TOK, question=_Q,
                buy_price=0.60, shares=100.0,
            )
            await temp_db.close_mm_round(rid, sell_price=0.62, realized_pnl=pnl, rebate_est=rebate)

        daily = await temp_db.get_daily_mm_pnl()
        assert abs(daily - 2.85) < 0.001  # gross 2.5 + rebates 0.35

    @pytest.mark.asyncio
    async def test_excludes_open_rounds(self, temp_db: Database):
        """Open rounds (not closed) should not be counted."""
        await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        daily = await temp_db.get_daily_mm_pnl()
        assert daily == 0.0

    @pytest.mark.asyncio
    async def test_no_rounds_returns_zero(self, temp_db: Database):
        daily = await temp_db.get_daily_mm_pnl()
        assert daily == 0.0


class TestGetActiveMMRounds:
    """get_active_mm_rounds() returns only BUY_POSTED/BOUGHT/SELL_POSTED."""

    @pytest.mark.asyncio
    async def test_includes_buy_posted(self, temp_db: Database):
        await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        active = await temp_db.get_active_mm_rounds()
        assert len(active) == 1
        assert active[0]["status"] == "BUY_POSTED"

    @pytest.mark.asyncio
    async def test_includes_bought(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.update_mm_round_bought(rid)
        active = await temp_db.get_active_mm_rounds()
        assert active[0]["status"] == "BOUGHT"

    @pytest.mark.asyncio
    async def test_includes_sell_posted(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.update_mm_round_bought(rid)
        await temp_db.update_mm_round_sell_posted(rid, sell_price=0.62, sell_order_id="s1")
        active = await temp_db.get_active_mm_rounds()
        assert active[0]["status"] == "SELL_POSTED"

    @pytest.mark.asyncio
    async def test_excludes_closed(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.close_mm_round(rid, sell_price=0.62, realized_pnl=2.0)
        active = await temp_db.get_active_mm_rounds()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_excludes_cancelled(self, temp_db: Database):
        rid = await temp_db.insert_mm_round(
            market_id=_MKT, token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.cancel_mm_round(rid)
        active = await temp_db.get_active_mm_rounds()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_mixed_statuses(self, temp_db: Database):
        """Only active statuses should appear."""
        # One active
        await temp_db.insert_mm_round(
            market_id="mkt_1", token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        # One closed
        r2 = await temp_db.insert_mm_round(
            market_id="mkt_2", token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.close_mm_round(r2, sell_price=0.62, realized_pnl=2.0)

        # One cancelled
        r3 = await temp_db.insert_mm_round(
            market_id="mkt_3", token_id=_TOK, question=_Q,
            buy_price=0.60, shares=100.0,
        )
        await temp_db.cancel_mm_round(r3)

        active = await temp_db.get_active_mm_rounds()
        assert len(active) == 1
        assert active[0]["market_id"] == "mkt_1"
