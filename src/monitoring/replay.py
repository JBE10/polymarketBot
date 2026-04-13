from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.storage.database import Database


@dataclass
class ReplaySummary:
    total: int
    matched: int
    mismatched: int
    mismatch_rate: float


class DecisionReplayService:
    """Deterministic replay checker based on persisted decision snapshots."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def _recompute_edge_net_pct(row: dict[str, Any]) -> float | None:
        p_model_up = row.get("p_model_up")
        p_market_up = row.get("p_market_up")
        total_cost_pct = row.get("total_cost_pct")
        side = row.get("side")

        if p_model_up is None or p_market_up is None or total_cost_pct is None:
            return None

        if side == "UP":
            edge_raw = p_model_up - p_market_up
        elif side == "DOWN":
            edge_raw = (1.0 - p_model_up) - (1.0 - p_market_up)
        else:
            return None

        return (edge_raw - (total_cost_pct / 100.0)) * 100.0

    @staticmethod
    def _snapshot_hash(row: dict[str, Any]) -> str:
        payload = {
            "decision_ts": row.get("decision_ts"),
            "asset": row.get("asset"),
            "market_id": row.get("market_id"),
            "side": row.get("side"),
            "action": row.get("action"),
            "reason_code": row.get("reason_code"),
            "p_model_up": row.get("p_model_up"),
            "p_market_up": row.get("p_market_up"),
            "edge_net_pct": row.get("edge_net_pct"),
            "total_cost_pct": row.get("total_cost_pct"),
            "score": row.get("score"),
        }
        packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()

    async def replay_day(self, *, date_utc: str, tolerance_pct: float = 1e-6) -> dict[str, Any]:
        rows = await self._db.get_decision_snapshots(
            start_ts=f"{date_utc}T00:00:00Z",
            end_ts=f"{date_utc}T23:59:59Z",
            limit=100000,
        )

        mismatches: list[dict[str, Any]] = []
        matched = 0

        for row in rows:
            expected = self._recompute_edge_net_pct(row)
            actual = row.get("edge_net_pct")
            row_hash = self._snapshot_hash(row)

            if expected is None or actual is None:
                matched += 1
                continue

            delta = abs(expected - actual)
            if delta > tolerance_pct:
                mismatches.append(
                    {
                        "id": row.get("id"),
                        "asset": row.get("asset"),
                        "market_id": row.get("market_id"),
                        "expected_edge_net_pct": expected,
                        "actual_edge_net_pct": actual,
                        "delta_pct": delta,
                        "snapshot_hash": row_hash,
                    }
                )
            else:
                matched += 1

        total = len(rows)
        summary = ReplaySummary(
            total=total,
            matched=matched,
            mismatched=len(mismatches),
            mismatch_rate=(len(mismatches) / total) if total else 0.0,
        )

        return {
            "date_utc": date_utc,
            "summary": {
                "total": summary.total,
                "matched": summary.matched,
                "mismatched": summary.mismatched,
                "mismatch_rate": summary.mismatch_rate,
            },
            "mismatches": mismatches,
        }
