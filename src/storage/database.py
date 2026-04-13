"""
Async SQLite persistence using aiosqlite with WAL journaling.
Stores: orders, positions, LLM evaluations, and RAG ingest log.
"""
from __future__ import annotations

import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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

CREATE TABLE IF NOT EXISTS decision_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    asset                TEXT    NOT NULL,
    market_id            TEXT    NOT NULL,
    side                 TEXT    NOT NULL,       -- UP | DOWN | N/A
    action               TEXT    NOT NULL,       -- BUY | SKIP
    reason_code          TEXT    NOT NULL,
    regime               TEXT    NOT NULL,       -- LOW_VOL | MID_VOL | HIGH_VOL
    p_model_up           REAL,
    p_market_up          REAL,
    edge_net_pct         REAL,
    score                REAL,
    threshold_pct        REAL,
    total_cost_pct       REAL,
    fee_pct              REAL,
    slippage_pct         REAL,
    latency_buffer_pct   REAL,
    notional_usd         REAL,
    spread_pct           REAL,
    depth_usd            REAL,
    expiry_utc           TEXT,
    feature_ema_fast     REAL,
    feature_ema_slow     REAL,
    feature_rsi          REAL,
    feature_momentum     REAL,
    feature_atr_pctile   REAL,
    meta_json            TEXT
);

CREATE TABLE IF NOT EXISTS operational_incidents (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc               TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    source               TEXT    NOT NULL,
    severity             TEXT    NOT NULL,       -- INFO | WARN | SEVERE
    incident_type        TEXT    NOT NULL,
    message              TEXT    NOT NULL,
    details_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_market      ON orders(market_id);
CREATE INDEX IF NOT EXISTS idx_positions_market   ON positions(market_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_mkt    ON evaluations(market_id, created_at);
CREATE INDEX IF NOT EXISTS idx_mm_rounds_market   ON mm_rounds(market_id, status);
CREATE INDEX IF NOT EXISTS idx_mm_fills_market    ON mm_fills(market_id, filled_at);
CREATE INDEX IF NOT EXISTS idx_calibration_market ON llm_calibration(market_id);
CREATE INDEX IF NOT EXISTS idx_decision_ts        ON decision_snapshots(decision_ts);
CREATE INDEX IF NOT EXISTS idx_decision_asset_ts  ON decision_snapshots(asset, decision_ts);
CREATE INDEX IF NOT EXISTS idx_decision_market_ts ON decision_snapshots(market_id, decision_ts);
CREATE INDEX IF NOT EXISTS idx_incidents_ts        ON operational_incidents(ts_utc);
CREATE INDEX IF NOT EXISTS idx_incidents_severity  ON operational_incidents(severity, ts_utc);
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
        await self._db.commit()
        log.info("Database ready at %s", self.path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            log.info("Database connection closed.")

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
        cost = None
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
    ) -> int:
        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO evaluations
                    (market_id, question, market_price, estimated_prob, expected_value,
                     kelly_fraction, position_size_usd, confidence, reasoning,
                     key_factors, action, skip_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, question, market_price, estimated_prob, expected_value,
                    kelly_fraction, position_size_usd, confidence.upper(), reasoning,
                    key_factors, action.upper(), skip_reason,
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def was_recently_evaluated(
        self, market_id: str, within_hours: int = 6
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
        async with self._tx() as db:
            await db.execute(
                """
                UPDATE mm_rounds
                SET status='CLOSED', sell_price=?, realized_pnl=?,
                    rebate_est=?,
                    closed_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id=?
                """,
                (sell_price, realized_pnl, rebate_est, round_id),
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
            SELECT realized_pnl FROM mm_rounds
            WHERE market_id = ? AND status = 'CLOSED'
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (market_id, n),
        ) as cur:
            rows = await cur.fetchall()

        streak = 0
        for row in rows:
            if (row["realized_pnl"] or 0) < 0:
                streak += 1
            else:
                break
        return streak

    async def get_daily_mm_pnl(self) -> float:
        """Sum of realized P&L from mm_rounds closed today."""
        assert self._db
        async with self._db.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0) AS pnl
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
        toxic = sum(1 for f in fills if (f.get("adverse_move") or 0) >= 0.005)
        return toxic / len(fills)

    async def get_mm_closed_rounds(
        self,
        *,
        start_ts: str | None = None,
        end_ts: str | None = None,
        limit: int = 100000,
    ) -> list[dict[str, Any]]:
        assert self._db
        query = """
            SELECT id, market_id, realized_pnl, rebate_est, closed_at
            FROM mm_rounds
            WHERE status = 'CLOSED'
        """
        params: list[Any] = []

        if start_ts:
            query += " AND closed_at >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND closed_at <= ?"
            params.append(end_ts)

        query += " ORDER BY closed_at ASC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, tuple(params)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_mm_performance_summary(
        self,
        *,
        start_ts: str | None = None,
        end_ts: str | None = None,
    ) -> dict[str, float | int]:
        rounds = await self.get_mm_closed_rounds(start_ts=start_ts, end_ts=end_ts)
        pnl_values = [float((r.get("realized_pnl") or 0.0) + (r.get("rebate_est") or 0.0)) for r in rounds]

        wins = sum(1 for p in pnl_values if p > 0)
        losses = sum(1 for p in pnl_values if p < 0)
        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = abs(sum(p for p in pnl_values if p < 0))
        net_pnl = sum(pnl_values)
        total = len(pnl_values)

        return {
            "trades": total,
            "wins": wins,
            "losses": losses,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": net_pnl,
            "ev_per_trade": (net_pnl / total) if total else 0.0,
        }

    async def get_mm_market_stats(
        self,
        market_id: str,
        *,
        lookback: int = 12,
    ) -> dict[str, float | int]:
        """Return recent per-market MM summary for blacklist and diagnostics."""
        assert self._db
        async with self._db.execute(
            """
            SELECT realized_pnl, rebate_est
            FROM mm_rounds
            WHERE status = 'CLOSED' AND market_id = ?
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (market_id, lookback),
        ) as cur:
            rows = await cur.fetchall()

        pnl_values = [float((r["realized_pnl"] or 0.0) + (r["rebate_est"] or 0.0)) for r in rows]
        trades = len(pnl_values)
        wins = sum(1 for p in pnl_values if p > 0)
        losses = sum(1 for p in pnl_values if p < 0)
        net_pnl = sum(pnl_values)

        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "net_pnl": net_pnl,
            "win_rate": (wins / trades) if trades else 0.0,
        }

    # ── Decision snapshots (P4 replay/observability) ─────────────────────

    async def insert_decision_snapshot(
        self,
        *,
        decision_ts: str | None = None,
        asset: str,
        market_id: str,
        side: str,
        action: str,
        reason_code: str,
        regime: str,
        p_model_up: float | None,
        p_market_up: float | None,
        edge_net_pct: float | None,
        score: float | None,
        threshold_pct: float | None,
        total_cost_pct: float | None,
        fee_pct: float | None,
        slippage_pct: float | None,
        latency_buffer_pct: float | None,
        notional_usd: float | None,
        spread_pct: float | None,
        depth_usd: float | None,
        expiry_utc: str | None,
        feature_ema_fast: float | None,
        feature_ema_slow: float | None,
        feature_rsi: float | None,
        feature_momentum: float | None,
        feature_atr_pctile: float | None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO decision_snapshots (
                    decision_ts, asset, market_id, side, action, reason_code, regime,
                    p_model_up, p_market_up, edge_net_pct, score,
                    threshold_pct, total_cost_pct, fee_pct, slippage_pct, latency_buffer_pct,
                    notional_usd, spread_pct, depth_usd, expiry_utc,
                    feature_ema_fast, feature_ema_slow, feature_rsi, feature_momentum,
                    feature_atr_pctile, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    asset,
                    market_id,
                    side,
                    action,
                    reason_code,
                    regime,
                    p_model_up,
                    p_market_up,
                    edge_net_pct,
                    score,
                    threshold_pct,
                    total_cost_pct,
                    fee_pct,
                    slippage_pct,
                    latency_buffer_pct,
                    notional_usd,
                    spread_pct,
                    depth_usd,
                    expiry_utc,
                    feature_ema_fast,
                    feature_ema_slow,
                    feature_rsi,
                    feature_momentum,
                    feature_atr_pctile,
                    json.dumps(meta or {}, sort_keys=True),
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def get_decision_snapshots(
        self,
        *,
        start_ts: str | None = None,
        end_ts: str | None = None,
        asset: str | None = None,
        market_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        assert self._db
        query = "SELECT * FROM decision_snapshots WHERE 1=1"
        params: list[Any] = []

        if start_ts:
            query += " AND decision_ts >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND decision_ts <= ?"
            params.append(end_ts)
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if market_id:
            query += " AND market_id = ?"
            params.append(market_id)

        query += " ORDER BY decision_ts DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, tuple(params)) as cur:
            rows = await cur.fetchall()
        out = [dict(r) for r in rows]
        for row in out:
            raw_meta = row.get("meta_json")
            try:
                row["meta"] = json.loads(raw_meta) if raw_meta else {}
            except Exception:
                row["meta"] = {}
        return out

    async def get_daily_decision_report(
        self,
        *,
        date_utc: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return daily aggregates by asset/regime/hour/side for observability."""
        assert self._db
        target_date = date_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        by_asset_sql = """
            SELECT
                asset,
                COUNT(*) AS total_decisions,
                SUM(CASE WHEN action='BUY' THEN 1 ELSE 0 END) AS buy_decisions,
                AVG(COALESCE(edge_net_pct, 0)) AS avg_edge_net_pct,
                AVG(COALESCE(score, 0)) AS avg_score,
                SUM(COALESCE(notional_usd, 0)) AS total_notional_usd
            FROM decision_snapshots
            WHERE date(decision_ts) = date(?)
            GROUP BY asset
            ORDER BY asset
        """

        by_regime_sql = """
            SELECT
                regime,
                COUNT(*) AS total_decisions,
                SUM(CASE WHEN action='BUY' THEN 1 ELSE 0 END) AS buy_decisions,
                AVG(COALESCE(edge_net_pct, 0)) AS avg_edge_net_pct
            FROM decision_snapshots
            WHERE date(decision_ts) = date(?)
            GROUP BY regime
            ORDER BY regime
        """

        by_hour_sql = """
            SELECT
                strftime('%H', decision_ts) AS hour_utc,
                COUNT(*) AS total_decisions,
                SUM(CASE WHEN action='BUY' THEN 1 ELSE 0 END) AS buy_decisions,
                AVG(COALESCE(edge_net_pct, 0)) AS avg_edge_net_pct
            FROM decision_snapshots
            WHERE date(decision_ts) = date(?)
            GROUP BY hour_utc
            ORDER BY hour_utc
        """

        by_side_sql = """
            SELECT
                side,
                COUNT(*) AS total_decisions,
                SUM(CASE WHEN action='BUY' THEN 1 ELSE 0 END) AS buy_decisions,
                AVG(COALESCE(edge_net_pct, 0)) AS avg_edge_net_pct
            FROM decision_snapshots
            WHERE date(decision_ts) = date(?)
            GROUP BY side
            ORDER BY side
        """

        async with self._db.execute(by_asset_sql, (target_date,)) as cur:
            by_asset = [dict(r) for r in await cur.fetchall()]
        async with self._db.execute(by_regime_sql, (target_date,)) as cur:
            by_regime = [dict(r) for r in await cur.fetchall()]
        async with self._db.execute(by_hour_sql, (target_date,)) as cur:
            by_hour = [dict(r) for r in await cur.fetchall()]
        async with self._db.execute(by_side_sql, (target_date,)) as cur:
            by_side = [dict(r) for r in await cur.fetchall()]

        return {
            "date_utc": [{"date": target_date}],
            "by_asset": by_asset,
            "by_regime": by_regime,
            "by_hour": by_hour,
            "by_side": by_side,
        }

    # ── Operational incidents (P5 promotion gate) ─────────────────────────

    async def insert_operational_incident(
        self,
        *,
        source: str,
        severity: str,
        incident_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        severity_upper = severity.upper()
        if severity_upper not in {"INFO", "WARN", "SEVERE"}:
            raise ValueError("severity must be INFO, WARN, or SEVERE")

        async with self._tx() as db:
            cur = await db.execute(
                """
                INSERT INTO operational_incidents
                    (source, severity, incident_type, message, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source,
                    severity_upper,
                    incident_type,
                    message,
                    json.dumps(details or {}, sort_keys=True),
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    async def get_operational_incidents(
        self,
        *,
        start_ts: str | None = None,
        end_ts: str | None = None,
        severity: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        assert self._db
        query = "SELECT * FROM operational_incidents WHERE 1=1"
        params: list[Any] = []

        if start_ts:
            query += " AND ts_utc >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND ts_utc <= ?"
            params.append(end_ts)
        if severity:
            query += " AND severity = ?"
            params.append(severity.upper())

        query += " ORDER BY ts_utc DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, tuple(params)) as cur:
            rows = await cur.fetchall()
        out = [dict(r) for r in rows]
        for row in out:
            raw = row.get("details_json")
            try:
                row["details"] = json.loads(raw) if raw else {}
            except Exception:
                row["details"] = {}
        return out
