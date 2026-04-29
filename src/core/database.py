"""
Async SQLite persistence using aiosqlite with WAL journaling.
Stores: orders, positions, LLM evaluations, and RAG ingest log.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import aiosqlite

log = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS orders (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    polymarket_order_id  TEXT    UNIQUE,
    market_id            TEXT    NOT NULL,
    token_id             TEXT    NOT NULL,
    question             TEXT    NOT NULL,
    side                 TEXT    NOT NULL,       -- BUY | SELL
    price                REAL    NOT NULL,
    size                 REAL    NOT NULL,
    status               TEXT    NOT NULL DEFAULT 'PENDING',
    transaction_hash     TEXT,
    error_message        TEXT,
    created_at           TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at           TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS positions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id            TEXT    NOT NULL UNIQUE,
    token_id             TEXT    NOT NULL,
    question             TEXT    NOT NULL,
    outcome              TEXT    NOT NULL,       -- YES | NO
    avg_entry_price      REAL    NOT NULL,
    current_price        REAL,
    shares               REAL    NOT NULL,
    cost_usd             REAL    NOT NULL,
    current_value_usd    REAL,
    unrealized_pnl       REAL,
    status               TEXT    NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED
    created_at           TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at           TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS evaluations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id            TEXT    NOT NULL,
    question             TEXT    NOT NULL,
    market_price         REAL    NOT NULL,
    estimated_prob       REAL    NOT NULL,
    expected_value       REAL    NOT NULL,
    kelly_fraction       REAL    NOT NULL,
    position_size_usd    REAL    NOT NULL,
    chosen_side          TEXT,
    side_price           REAL,
    no_price             REAL,
    mc_samples           INTEGER,
    mc_mean_edge         REAL,
    mc_p05_edge          REAL,
    mc_p95_edge          REAL,
    confidence           TEXT    NOT NULL,       -- LOW | MEDIUM | HIGH
    reasoning            TEXT,
    key_factors          TEXT,                   -- JSON array
    action               TEXT    NOT NULL,       -- BUY | SKIP | REJECT
    skip_reason          TEXT,
    created_at           TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    source               TEXT    NOT NULL,
    title                TEXT,
    content_hash         TEXT    UNIQUE,
    doc_id               TEXT,
    ingested_at          TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS mm_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id       TEXT    NOT NULL,
    token_id        TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    buy_price       REAL    NOT NULL,
    buy_order_id    TEXT,
    sell_price      REAL,
    sell_order_id   TEXT,
    shares          REAL    NOT NULL,
    realized_pnl    REAL,
    rebate_est      REAL    DEFAULT 0,
    net_pnl         REAL,
    status          TEXT    NOT NULL DEFAULT 'BUY_POSTED',
    created_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    closed_at       TEXT
);

CREATE TABLE IF NOT EXISTS llm_calibration (
    market_id       TEXT    PRIMARY KEY,
    question        TEXT    NOT NULL,
    predicted_prob  REAL    NOT NULL,
    market_price    REAL    NOT NULL,
    llm_provider    TEXT    NOT NULL DEFAULT 'unknown',
    resolved_yes    INTEGER,
    created_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    resolved_at     TEXT
);

CREATE TABLE IF NOT EXISTS mm_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id        INTEGER NOT NULL,
    market_id       TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    fill_price      REAL    NOT NULL,
    mid_price_after REAL,
    adverse_move    REAL,
    filled_at       TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_market      ON orders(market_id);
CREATE INDEX IF NOT EXISTS idx_positions_market   ON positions(market_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_mkt    ON evaluations(market_id, created_at);
CREATE INDEX IF NOT EXISTS idx_mm_rounds_market   ON mm_rounds(market_id, status);
CREATE INDEX IF NOT EXISTS idx_mm_fills_market    ON mm_fills(market_id, filled_at);
CREATE INDEX IF NOT EXISTS idx_calibration_market ON llm_calibration(market_id);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate_schema()
        await self._db.commit()
        log.info("Database ready at %s", self.path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            log.info("Database connection closed.")

    async def _migrate_schema(self) -> None:
        """Apply lightweight migrations for existing SQLite databases."""
        assert self._db
        async with self._db.execute("PRAGMA table_info(mm_rounds)") as cur:
            columns = {row["name"] for row in await cur.fetchall()}
        if "net_pnl" not in columns:
            await self._db.execute("ALTER TABLE mm_rounds ADD COLUMN net_pnl REAL")
            await self._db.execute(
                """
                UPDATE mm_rounds
                SET net_pnl = COALESCE(realized_pnl, 0) + COALESCE(rebate_est, 0)
                WHERE status = 'CLOSED'
                  AND realized_pnl IS NOT NULL
                """
            )

        async with self._db.execute("PRAGMA table_info(evaluations)") as cur:
            eval_columns = {row["name"] for row in await cur.fetchall()}
        eval_migrations = {
            "chosen_side": "TEXT",
            "side_price": "REAL",
            "no_price": "REAL",
            "mc_samples": "INTEGER",
            "mc_mean_edge": "REAL",
            "mc_p05_edge": "REAL",
            "mc_p95_edge": "REAL",
        }
        for column, col_type in eval_migrations.items():
            if column not in eval_columns:
                await self._db.execute(
                    f"ALTER TABLE evaluations ADD COLUMN {column} {col_type}"
                )

    @asynccontextmanager
    async def _tx(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Context manager that commits on success and rolls back on error."""
        assert self._db, "Database not connected — call connect() first."
        try:
            yield self._db
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    # ── Orders ────────────────────────────────────────────────────────────────

    async def insert_order(
        self,
        market_id: str,
        token_id: str,
        question: str,
        side: str,
        price: float,
        size: float,
        status: str = "PENDING",
        polymarket_order_id: str | None = None,
        transaction_hash: str | None = None,
    ) -> int:
        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO orders
                    (polymarket_order_id, market_id, token_id, question,
                     side, price, size, status, transaction_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    polymarket_order_id, market_id, token_id, question,
                    side.upper(), price, size, status.upper(), transaction_hash,
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def update_order_status(
        self,
        order_id: int,
        status: str,
        transaction_hash: str | None = None,
        error_message: str | None = None,
    ) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                UPDATE orders
                SET status = ?, transaction_hash = ?, error_message = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE id = ?
                """,
                (status.upper(), transaction_hash, error_message, order_id),
            )

    async def get_recent_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        assert self._db
        async with self._db.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Positions ─────────────────────────────────────────────────────────────

    async def upsert_position(
        self,
        market_id: str,
        token_id: str,
        question: str,
        outcome: str,
        avg_entry_price: float,
        shares: float,
        cost_usd: float,
        current_price: float | None = None,
    ) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                INSERT INTO positions
                    (market_id, token_id, question, outcome,
                     avg_entry_price, shares, cost_usd, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    avg_entry_price = excluded.avg_entry_price,
                    shares          = excluded.shares,
                    cost_usd        = excluded.cost_usd,
                    current_price   = excluded.current_price,
                    updated_at      = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                """,
                (
                    market_id, token_id, question, outcome.upper(),
                    avg_entry_price, shares, cost_usd, current_price,
                ),
            )

    async def get_open_positions(self) -> list[dict[str, Any]]:
        assert self._db
        async with self._db.execute(
            "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def close_position(
        self,
        market_id: str,
        exit_price: float | None = None,
        exit_reason: str = "",
    ) -> None:
        """Mark a position as CLOSED and record realized P&L."""
        async with self._tx() as db:
            # Fetch shares and cost to compute realized P&L
            async with db.execute(
                "SELECT shares, cost_usd FROM positions WHERE market_id = ?",
                (market_id,),
            ) as cur:
                row = await cur.fetchone()

            realized_pnl: float | None = None
            if row and exit_price is not None:
                realized_pnl = row["shares"] * exit_price - row["cost_usd"]

            await db.execute(
                """
                UPDATE positions
                SET status        = 'CLOSED',
                    current_price = COALESCE(?, current_price),
                    unrealized_pnl = NULL,
                    current_value_usd = COALESCE(
                        CASE WHEN ? IS NOT NULL THEN shares * ? ELSE NULL END,
                        current_value_usd
                    ),
                    updated_at    = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE market_id = ?
                """,
                (exit_price, exit_price, exit_price, market_id),
            )
            # Store realized P&L in a note field if available; log it here
            if realized_pnl is not None:
                log.info(
                    "Position CLOSED [%s]: exit=%.3f  realized_pnl=%+.2f  reason=%s",
                    market_id[:10], exit_price, realized_pnl, exit_reason or "manual",
                )

    async def update_position_price(
        self, market_id: str, current_price: float
    ) -> None:
        current_value = None
        async with self._tx() as db:
            async with db.execute(
                "SELECT shares, cost_usd FROM positions WHERE market_id = ?",
                (market_id,),
            ) as cur:
                row = await cur.fetchone()
            if row:
                current_value = row["shares"] * current_price
                pnl = current_value - row["cost_usd"]
                await db.execute(
                    """
                    UPDATE positions
                    SET current_price=?, current_value_usd=?, unrealized_pnl=?,
                        updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    WHERE market_id=?
                    """,
                    (current_price, current_value, pnl, market_id),
                )

    # ── Evaluations ───────────────────────────────────────────────────────────

    async def insert_evaluation(
        self,
        market_id: str,
        question: str,
        market_price: float,
        estimated_prob: float,
        expected_value: float,
        kelly_fraction: float,
        position_size_usd: float,
        confidence: str,
        reasoning: str,
        key_factors: str,   # JSON-serialised list
        action: str,
        skip_reason: str = "",
        chosen_side: str | None = None,
        side_price: float | None = None,
        no_price: float | None = None,
        mc_samples: int | None = None,
        mc_mean_edge: float | None = None,
        mc_p05_edge: float | None = None,
        mc_p95_edge: float | None = None,
    ) -> int:
        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO evaluations
                    (market_id, question, market_price, estimated_prob, expected_value,
                     kelly_fraction, position_size_usd, chosen_side, side_price,
                     no_price, mc_samples, mc_mean_edge, mc_p05_edge, mc_p95_edge,
                     confidence, reasoning, key_factors, action, skip_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, question, market_price, estimated_prob, expected_value,
                    kelly_fraction, position_size_usd,
                    chosen_side.upper() if chosen_side else None,
                    side_price, no_price, mc_samples, mc_mean_edge, mc_p05_edge,
                    mc_p95_edge, confidence.upper(), reasoning, key_factors,
                    action.upper(), skip_reason,
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def was_recently_evaluated(
        self, market_id: str, within_hours: float = 6
    ) -> bool:
        """True if this market was already evaluated in the last N hours."""
        assert self._db
        async with self._db.execute(
            """
            SELECT 1 FROM evaluations
            WHERE market_id = ?
              AND created_at >= datetime('now', ? || ' hours')
            LIMIT 1
            """,
            (market_id, f"-{within_hours}"),
        ) as cur:
            return await cur.fetchone() is not None

    # ── Ingest log ────────────────────────────────────────────────────────────

    async def log_ingest(
        self,
        source: str,
        title: str = "",
        content_hash: str = "",
        doc_id: str = "",
    ) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO ingest_log (source, title, content_hash, doc_id)
                VALUES (?, ?, ?, ?)
                """,
                (source, title, content_hash, doc_id),
            )

    async def is_already_ingested(self, content_hash: str) -> bool:
        assert self._db
        async with self._db.execute(
            "SELECT 1 FROM ingest_log WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ) as cur:
            return await cur.fetchone() is not None

    # ── Market-Making Rounds ───────────────────────────────────────────────

    async def insert_mm_round(
        self,
        market_id: str,
        token_id: str,
        question: str,
        buy_price: float,
        shares: float,
        buy_order_id: str | None = None,
    ) -> int:
        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO mm_rounds
                    (market_id, token_id, question, buy_price, shares,
                     buy_order_id, status)
                VALUES (?, ?, ?, ?, ?, ?, 'BUY_POSTED')
                """,
                (market_id, token_id, question, buy_price, shares, buy_order_id),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def update_mm_round_bought(self, round_id: int) -> None:
        """Transition from BUY_POSTED → BOUGHT after buy fill detected."""
        async with self._tx() as db:
            await db.execute(
                "UPDATE mm_rounds SET status='BOUGHT' WHERE id=?", (round_id,)
            )

    async def update_mm_round_sell_posted(
        self, round_id: int, sell_price: float, sell_order_id: str | None = None
    ) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                UPDATE mm_rounds
                SET status='SELL_POSTED', sell_price=?, sell_order_id=?
                WHERE id=?
                """,
                (sell_price, sell_order_id, round_id),
            )

    async def close_mm_round(
        self,
        round_id: int,
        sell_price: float,
        realized_pnl: float,
        rebate_est: float = 0.0,
    ) -> None:
        net_pnl = realized_pnl + rebate_est
        async with self._tx() as db:
            await db.execute(
                """
                UPDATE mm_rounds
                SET status='CLOSED', sell_price=?, realized_pnl=?,
                    rebate_est=?, net_pnl=?,
                    closed_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id=?
                """,
                (sell_price, realized_pnl, rebate_est, net_pnl, round_id),
            )

    async def cancel_mm_round(self, round_id: int) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                UPDATE mm_rounds
                SET status='CANCELLED',
                    closed_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id=?
                """,
                (round_id,),
            )

    async def get_active_mm_rounds(self) -> list[dict[str, Any]]:
        """All rounds that are not CLOSED or CANCELLED."""
        assert self._db
        async with self._db.execute(
            """
            SELECT * FROM mm_rounds
            WHERE status IN ('BUY_POSTED', 'BOUGHT', 'SELL_POSTED')
            ORDER BY created_at DESC
            """
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_consecutive_losses(self, market_id: str, n: int = 10) -> int:
        """Count unbroken trailing losses for a market (most recent first)."""
        assert self._db
        async with self._db.execute(
            """
            SELECT COALESCE(net_pnl, realized_pnl + COALESCE(rebate_est, 0), realized_pnl) AS pnl
            FROM mm_rounds
            WHERE market_id = ? AND status = 'CLOSED'
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (market_id, n),
        ) as cur:
            rows = await cur.fetchall()

        streak = 0
        for row in rows:
            if (row["pnl"] or 0) < 0:
                streak += 1
            else:
                break
        return streak

    async def get_daily_mm_pnl(self) -> float:
        """Sum of net P&L from mm_rounds closed today."""
        assert self._db
        async with self._db.execute(
            """
            SELECT COALESCE(SUM(
                COALESCE(net_pnl, realized_pnl + COALESCE(rebate_est, 0), realized_pnl)
            ), 0) AS pnl
            FROM mm_rounds
            WHERE status = 'CLOSED'
              AND closed_at >= date('now')
            """
        ) as cur:
            row = await cur.fetchone()
        return float(row["pnl"]) if row else 0.0

    # ── LLM Calibration ───────────────────────────────────────────────────

    async def upsert_calibration(
        self,
        market_id: str,
        question: str,
        predicted_prob: float,
        market_price: float,
        llm_provider: str = "unknown",
    ) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                INSERT INTO llm_calibration
                    (market_id, question, predicted_prob, market_price, llm_provider)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    predicted_prob = excluded.predicted_prob,
                    market_price   = excluded.market_price,
                    llm_provider   = excluded.llm_provider
                """,
                (market_id, question, predicted_prob, market_price, llm_provider),
            )

    async def update_calibration_resolved(
        self, market_id: str, resolved_yes: bool
    ) -> None:
        async with self._tx() as db:
            await db.execute(
                """
                UPDATE llm_calibration
                SET resolved_yes = ?, resolved_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE market_id = ?
                """,
                (1 if resolved_yes else 0, market_id),
            )

    async def get_calibration_stats(self) -> dict[str, Any]:
        """Return calibration summary for resolved markets."""
        assert self._db
        async with self._db.execute(
            """
            SELECT
                COUNT(*) as total,
                AVG(predicted_prob) as avg_predicted,
                AVG(resolved_yes) as avg_resolved,
                AVG(ABS(predicted_prob - market_price)) as avg_deviation
            FROM llm_calibration
            WHERE resolved_yes IS NOT NULL
            """
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else {}

    # ── MM Fills (toxicity tracking) ───────────────────────────────────────

    async def insert_mm_fill(
        self,
        round_id: int,
        market_id: str,
        side: str,
        fill_price: float,
        mid_price_after: float | None = None,
    ) -> int:
        adverse = None
        if mid_price_after is not None:
            if side.upper() == "BUY":
                adverse = fill_price - mid_price_after
            else:
                adverse = mid_price_after - fill_price

        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO mm_fills
                    (round_id, market_id, side, fill_price, mid_price_after, adverse_move)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (round_id, market_id, side.upper(), fill_price, mid_price_after, adverse),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def get_recent_mm_fills(
        self, market_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        assert self._db
        async with self._db.execute(
            """
            SELECT * FROM mm_fills
            WHERE market_id = ?
            ORDER BY filled_at DESC
            LIMIT ?
            """,
            (market_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_toxicity_ratio(self, market_id: str, lookback: int = 5) -> float:
        """Fraction of recent fills where price moved adversely (> 0)."""
        fills = await self.get_recent_mm_fills(market_id, lookback)
        if not fills:
            return 0.0
        toxic = sum(1 for f in fills if (f.get("adverse_move") or 0) > 0.005)
        return toxic / len(fills)
