"""
SQLite persistence layer.
Stores bookmarks, alerts, journal entries, wallet addresses, and API cache.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from config import DB_PATH

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS bookmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   TEXT    NOT NULL UNIQUE,
    question    TEXT    NOT NULL,
    token_id    TEXT,
    notes       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id    TEXT    NOT NULL,
    question     TEXT    NOT NULL,
    token_id     TEXT,
    outcome      TEXT    NOT NULL,   -- 'YES' | 'NO'
    condition    TEXT    NOT NULL,   -- 'above' | 'below'
    target_price REAL    NOT NULL,
    triggered    INTEGER DEFAULT 0,
    triggered_at TIMESTAMP,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   TEXT,
    question    TEXT    NOT NULL,
    trade_type  TEXT    NOT NULL,   -- 'BUY' | 'SELL' (observation only)
    outcome     TEXT    NOT NULL,   -- 'YES' | 'NO'
    price       REAL    NOT NULL,
    size_usd    REAL    NOT NULL,
    rationale   TEXT,
    result      TEXT,               -- 'WIN' | 'LOSS' | 'PENDING'
    pnl         REAL,
    tags        TEXT    DEFAULT '[]',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    address    TEXT NOT NULL UNIQUE,
    label      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init()

    # ── Connection helper ─────────────────────────────────────────────────────
    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Bookmarks ─────────────────────────────────────────────────────────────
    def add_bookmark(self, market_id: str, question: str,
                     token_id: str | None = None, notes: str | None = None) -> bool:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO bookmarks (market_id, question, token_id, notes) "
                    "VALUES (?, ?, ?, ?)",
                    (market_id, question, token_id, notes),
                )
            return True
        except Exception:
            return False

    def remove_bookmark(self, market_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM bookmarks WHERE market_id = ?", (market_id,))

    def get_bookmarks(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM bookmarks ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def is_bookmarked(self, market_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM bookmarks WHERE market_id = ?", (market_id,)
            ).fetchone()
        return row is not None

    def update_bookmark_notes(self, market_id: str, notes: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE bookmarks SET notes = ? WHERE market_id = ?", (notes, market_id)
            )

    # ── Alerts ────────────────────────────────────────────────────────────────
    def add_alert(self, market_id: str, question: str, token_id: str,
                  outcome: str, condition: str, target_price: float) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO alerts (market_id, question, token_id, outcome, condition, target_price) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (market_id, question, token_id, outcome.upper(), condition.lower(), target_price),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_alerts(self, active_only: bool = False) -> list[dict[str, Any]]:
        with self._conn() as conn:
            q = "SELECT * FROM alerts"
            if active_only:
                q += " WHERE triggered = 0"
            q += " ORDER BY created_at DESC"
            rows = conn.execute(q).fetchall()
        return [dict(r) for r in rows]

    def trigger_alert(self, alert_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE alerts SET triggered = 1, triggered_at = ? WHERE id = ?",
                (datetime.now().isoformat(), alert_id),
            )

    def delete_alert(self, alert_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))

    # ── Journal ───────────────────────────────────────────────────────────────
    def add_journal_entry(
        self,
        question: str,
        trade_type: str,
        outcome: str,
        price: float,
        size_usd: float,
        rationale: str = "",
        market_id: str | None = None,
        tags: list[str] | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO journal "
                "(market_id, question, trade_type, outcome, price, size_usd, rationale, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    market_id,
                    question,
                    trade_type.upper(),
                    outcome.upper(),
                    price,
                    size_usd,
                    rationale,
                    json.dumps(tags or []),
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_journal_entries(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM journal ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        entries = []
        for r in rows:
            e = dict(r)
            e["tags"] = json.loads(e.get("tags") or "[]")
            entries.append(e)
        return entries

    def resolve_journal_entry(
        self, entry_id: int, result: str, pnl: float | None = None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE journal SET result = ?, pnl = ?, resolved_at = ? WHERE id = ?",
                (result.upper(), pnl, datetime.now().isoformat(), entry_id),
            )

    def delete_journal_entry(self, entry_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM journal WHERE id = ?", (entry_id,))

    def get_journal_stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                            AS total,
                    SUM(CASE WHEN result='WIN'     THEN 1 ELSE 0 END)  AS wins,
                    SUM(CASE WHEN result='LOSS'    THEN 1 ELSE 0 END)  AS losses,
                    SUM(CASE WHEN result IS NULL
                              OR result='PENDING'  THEN 1 ELSE 0 END)  AS pending,
                    ROUND(SUM(COALESCE(pnl, 0)), 2)                    AS total_pnl,
                    ROUND(AVG(CASE WHEN pnl IS NOT NULL THEN pnl END), 2) AS avg_pnl
                FROM journal
                """
            ).fetchone()
        return dict(row) if row else {}

    # ── Wallets ───────────────────────────────────────────────────────────────
    def add_wallet(self, address: str, label: str = "") -> bool:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO wallets (address, label) VALUES (?, ?)",
                    (address.strip(), label.strip()),
                )
            return True
        except Exception:
            return False

    def get_wallets(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM wallets ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def remove_wallet(self, address: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM wallets WHERE address = ?", (address,))

    # ── API cache ─────────────────────────────────────────────────────────────
    def cache_set(self, key: str, value: Any, ttl: int = 60) -> None:
        expires = datetime.fromtimestamp(
            datetime.now().timestamp() + ttl
        ).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires),
            )

    def cache_get(self, key: str) -> Any | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM api_cache WHERE key = ?", (key,)
            ).fetchone()
        if row and datetime.fromisoformat(row["expires_at"]) > datetime.now():
            return json.loads(row["value"])
        return None

    def cache_purge(self) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM api_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            )


# Module-level singleton
db = Database()
