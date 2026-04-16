"""
In-memory cache for live order book state across all subscribed markets.

Updated by WsClient on every WebSocket push event.
Read by the Brain (analysis engine) and MarketMaker on every signal evaluation.

Thread-safety model
-------------------
All mutations go through a single asyncio.Lock so concurrent coroutines
(WsClient writer, Brain reader) never race.  The Lock is asyncio-native so
it never blocks the event loop.

Usage
-----
    cache = BookCache()
    # WsClient calls:
    await cache.update(token_id, bids, asks, timestamp)
    # Brain calls:
    snapshot = cache.get(token_id)   # returns BookSnapshot | None, non-blocking
    all_books = cache.get_all()      # dict[token_id, BookSnapshot]
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from src.polymarket.models import OrderBook, PriceLevel


@dataclass(frozen=True)
class BookSnapshot:
    """Immutable snapshot of a single token's order book at a point in time."""

    token_id: str
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    received_at: float  # time.monotonic() when the WS message arrived

    # ── Derived convenience properties ────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def age_seconds(self) -> float:
        """How old this snapshot is in seconds (monotonic clock)."""
        return time.monotonic() - self.received_at

    def to_order_book(self) -> OrderBook:
        """Convert to the existing OrderBook model for backward compatibility."""
        return OrderBook(
            market="",
            asset_id=self.token_id,
            bids=self.bids,
            asks=self.asks,
        )


class BookCache:
    """
    Shared, thread-safe store of order book snapshots.

    Designed for high-frequency reads (Brain polls on every WS event) and
    moderate writes (WsClient pushes on every book update message).

    Parameters
    ----------
    max_age_seconds : float
        Snapshots older than this are treated as stale (get() returns None).
        Default: 5.0 seconds — if the WS hasn't sent an update in 5s,
        something is wrong.
    """

    def __init__(self, max_age_seconds: float = 5.0) -> None:
        self._books: dict[str, BookSnapshot] = {}
        self._lock  = asyncio.Lock()
        self._max_age = max_age_seconds
        # Event fired after every update — Brain awaits this to wake up
        self._updated: asyncio.Event = asyncio.Event()

    # ── Write path (called by WsClient) ──────────────────────────────────────

    async def update(
        self,
        token_id: str,
        bids: list[dict],
        asks: list[dict],
    ) -> None:
        """
        Store a new order book snapshot for the given token.

        Bids/asks are raw dicts {"price": "0.45", "size": "100"} as
        received from the WebSocket.
        """
        parsed_bids = _parse_levels(bids, reverse=True)   # highest price first
        parsed_asks = _parse_levels(asks, reverse=False)   # lowest price first

        snapshot = BookSnapshot(
            token_id=token_id,
            bids=parsed_bids,
            asks=parsed_asks,
            received_at=time.monotonic(),
        )

        async with self._lock:
            self._books[token_id] = snapshot

        # Wake any coroutine waiting for an update (Brain, etc.)
        self._updated.set()
        self._updated.clear()

    async def remove(self, token_id: str) -> None:
        """Remove a token from the cache (called when unsubscribing)."""
        async with self._lock:
            self._books.pop(token_id, None)

    # ── Read path (called by Brain / MarketMaker) — non-blocking ─────────────

    def get(self, token_id: str) -> Optional[BookSnapshot]:
        """
        Return the latest snapshot for a token, or None if stale/missing.
        Non-blocking — no lock needed for reads because dict reads are
        atomic in CPython and we use immutable snapshots.
        """
        snapshot = self._books.get(token_id)
        if snapshot is None:
            return None
        if snapshot.age_seconds > self._max_age:
            return None  # too old — treat as missing
        return snapshot

    def get_all(self) -> dict[str, BookSnapshot]:
        """Return a shallow copy of all current snapshots."""
        return dict(self._books)

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """
        Convenience: return an OrderBook model from the cache.
        Compatible with existing code that expects OrderBook objects.
        Returns None if snapshot is missing or stale.
        """
        snap = self.get(token_id)
        return snap.to_order_book() if snap is not None else None

    @property
    def updated_event(self) -> asyncio.Event:
        """
        An asyncio.Event that fires after each update.
        Brain can await this to react immediately to new data:

            while True:
                await cache.updated_event.wait()
                process_new_books()
        """
        return self._updated

    def __len__(self) -> int:
        return len(self._books)

    def __repr__(self) -> str:
        return f"BookCache(tokens={len(self._books)}, max_age={self._max_age}s)"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_levels(raw: list[dict], *, reverse: bool) -> list[PriceLevel]:
    """Parse raw {"price": str, "size": str} dicts into PriceLevel objects."""
    levels = []
    for item in raw or []:
        try:
            price = float(item["price"])
            size  = float(item["size"])
            if size > 0:
                levels.append(PriceLevel(price=price, size=size))
        except (KeyError, ValueError, TypeError):
            continue
    levels.sort(key=lambda x: x.price, reverse=reverse)
    return levels
