"""
Phase 5 — Async SQLite WAL-mode database tests.

Uses the temp_db fixture for isolated testing. Validates CRUD operations,
WAL journal mode, and concurrent insert performance.
"""
from __future__ import annotations

import asyncio
from random import choice, random

import aiosqlite
import pytest

from src.core.database import Database


class TestDatabaseLifecycle:
    """Database connection, WAL mode, and schema creation."""

    @pytest.mark.asyncio
    async def test_wal_mode(self, temp_db: Database):
        """Database should use WAL journal mode for concurrent access."""
        async with aiosqlite.connect(temp_db.path) as conn:
            async with conn.execute("PRAGMA journal_mode") as cur:
                row = await cur.fetchone()
                journal_mode = row[0] if row else "unknown"
        assert journal_mode.lower() == "wal"

    @pytest.mark.asyncio
    async def test_tables_created(self, temp_db: Database):
        """Schema should create all required tables."""
        async with aiosqlite.connect(temp_db.path) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cur:
                rows = await cur.fetchall()
                tables = {row[0] for row in rows}

        expected = {"orders", "positions", "evaluations", "ingest_log", "mm_rounds", "mm_fills"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    @pytest.mark.asyncio
    async def test_mm_rounds_has_net_pnl_column(self, temp_db: Database):
        """Schema should persist net P&L separately from gross P&L and rebates."""
        async with aiosqlite.connect(temp_db.path) as conn:
            async with conn.execute("PRAGMA table_info(mm_rounds)") as cur:
                rows = await cur.fetchall()
                columns = {row[1] for row in rows}

        assert "net_pnl" in columns

    @pytest.mark.asyncio
    async def test_evaluations_has_short_term_decision_columns(self, temp_db: Database):
        """Schema should persist side selection and Monte Carlo diagnostics."""
        async with aiosqlite.connect(temp_db.path) as conn:
            async with conn.execute("PRAGMA table_info(evaluations)") as cur:
                rows = await cur.fetchall()
                columns = {row[1] for row in rows}

        assert {
            "chosen_side",
            "side_price",
            "no_price",
            "mc_samples",
            "mc_mean_edge",
            "mc_p05_edge",
            "mc_p95_edge",
        }.issubset(columns)


class TestEvaluations:
    """Evaluation insert and query operations."""

    @pytest.mark.asyncio
    async def test_insert_evaluation(self, temp_db: Database):
        row_id = await temp_db.insert_evaluation(
            market_id="mkt_001",
            question="Test market?",
            market_price=0.50,
            estimated_prob=0.70,
            expected_value=0.20,
            kelly_fraction=0.05,
            position_size_usd=50.0,
            confidence="HIGH",
            reasoning="Test reasoning",
            key_factors="[]",
            action="BUY",
        )
        assert row_id is not None
        assert row_id > 0

    @pytest.mark.asyncio
    async def test_was_recently_evaluated(self, temp_db: Database):
        await temp_db.insert_evaluation(
            market_id="mkt_002",
            question="Recent?",
            market_price=0.40,
            estimated_prob=0.60,
            expected_value=0.20,
            kelly_fraction=0.05,
            position_size_usd=25.0,
            confidence="MEDIUM",
            reasoning="Test",
            key_factors="[]",
            action="SKIP",
        )
        assert await temp_db.was_recently_evaluated("mkt_002", within_hours=1)
        assert not await temp_db.was_recently_evaluated("mkt_nonexistent")

    @pytest.mark.asyncio
    async def test_insert_short_term_decision_metadata(self, temp_db: Database):
        row_id = await temp_db.insert_evaluation(
            market_id="mkt_short",
            question="Will BTC go up in 15 minutes?",
            market_price=0.62,
            estimated_prob=0.40,
            expected_value=0.02,
            kelly_fraction=0.01,
            position_size_usd=10.0,
            confidence="HIGH",
            reasoning="NO has the better edge.",
            key_factors="[]",
            action="BUY",
            chosen_side="NO",
            side_price=0.38,
            no_price=0.38,
            mc_samples=2_000,
            mc_mean_edge=0.021,
            mc_p05_edge=-0.38,
            mc_p95_edge=0.62,
        )

        assert row_id > 0
        assert temp_db._db is not None
        async with temp_db._db.execute(
            """
            SELECT chosen_side, side_price, no_price, mc_samples, mc_mean_edge
            FROM evaluations
            WHERE id = ?
            """,
            (row_id,),
        ) as cur:
            row = await cur.fetchone()

        assert row["chosen_side"] == "NO"
        assert row["side_price"] == 0.38
        assert row["no_price"] == 0.38
        assert row["mc_samples"] == 2_000
        assert row["mc_mean_edge"] == 0.021


class TestOrders:
    """Order insert and status update operations."""

    @pytest.mark.asyncio
    async def test_insert_and_update_order(self, temp_db: Database):
        order_id = await temp_db.insert_order(
            market_id="mkt_003",
            token_id="tok_003",
            question="Order test?",
            side="BUY",
            price=0.55,
            size=100.0,
        )
        assert order_id > 0

        await temp_db.update_order_status(
            order_id=order_id,
            status="FILLED",
            transaction_hash="0x123abc",
        )

        orders = await temp_db.get_recent_orders(limit=1)
        assert len(orders) == 1
        assert orders[0]["status"] == "FILLED"


class TestPositions:
    """Position upsert, price update, and close operations."""

    @pytest.mark.asyncio
    async def test_upsert_and_query_position(self, temp_db: Database):
        await temp_db.upsert_position(
            market_id="mkt_004",
            token_id="tok_004",
            question="Position test?",
            outcome="YES",
            avg_entry_price=0.50,
            shares=200.0,
            cost_usd=100.0,
        )
        positions = await temp_db.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["market_id"] == "mkt_004"

    @pytest.mark.asyncio
    async def test_close_position(self, temp_db: Database):
        await temp_db.upsert_position(
            market_id="mkt_005",
            token_id="tok_005",
            question="Close test?",
            outcome="YES",
            avg_entry_price=0.50,
            shares=100.0,
            cost_usd=50.0,
        )
        await temp_db.close_position("mkt_005", exit_price=0.60)
        positions = await temp_db.get_open_positions()
        assert len(positions) == 0  # closed positions excluded


class TestConcurrentInserts:
    """Validate WAL mode handles concurrent writes without locking."""

    @pytest.mark.asyncio
    async def test_100_concurrent_inserts(self, temp_db: Database):
        """Insert 100 records concurrently and verify all are persisted."""
        tasks = [
            temp_db.insert_evaluation(
                market_id=f"bench_{i:04d}",
                question=f"Benchmark #{i}?",
                market_price=round(0.2 + random() * 0.6, 4),
                estimated_prob=round(random(), 4),
                expected_value=round(random() * 0.2 - 0.05, 4),
                kelly_fraction=round(random() * 0.15, 4),
                position_size_usd=round(random() * 200, 2),
                confidence=choice(["LOW", "MEDIUM", "HIGH"]),
                reasoning=f"Benchmark entry {i}",
                key_factors="[]",
                action=choice(["BUY", "SKIP"]),
            )
            for i in range(100)
        ]
        await asyncio.gather(*tasks)

        async with aiosqlite.connect(temp_db.path) as conn:
            async with conn.execute("SELECT COUNT(*) FROM evaluations") as cur:
                row = await cur.fetchone()
                count = row[0] if row else 0

        assert count == 100
