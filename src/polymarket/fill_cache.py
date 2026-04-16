"""
FillCache — in-memory registry of order fill events from the WebSocket user channel.

Updated by UserWsClient on every fill/cancel push event.
Read by MarketMaker to detect when a GTC order has been filled
WITHOUT calling get_open_orders() via HTTP.

Design
------
The CLOB user WebSocket pushes three event types we care about:

  "order"        — an order was accepted, partially filled, or cancelled
  "trade"        — a trade matched (both sides get this event)
  "error"        — auth or protocol error

For fill detection we track:
  - orders known to be OPEN (we posted them)
  - orders that were FILLED or CANCELLED (removed from open set)

The MarketMaker queries:
    cache.is_filled(order_id)      → bool
    cache.is_cancelled(order_id)   → bool
    cache.get_fill_price(order_id) → float | None
    cache.consume(order_id)        → removes from cache after reading

Thread-safety: asyncio.Lock protects all mutations.
Reads (is_filled, get_fill_price) are non-blocking dict lookups.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class FillStatus(str, Enum):
    OPEN      = "OPEN"       # order is live, not yet filled
    FILLED    = "FILLED"     # order fully matched
    PARTIAL   = "PARTIAL"    # partially filled (treated as open until fully done)
    CANCELLED = "CANCELLED"  # cancelled (by us or expired)
    UNKNOWN   = "UNKNOWN"    # event received but status unclear


@dataclass
class OrderEvent:
    """Snapshot of an order's latest known state from the WS user channel."""
    order_id:    str
    status:      FillStatus
    size_matched: float         # shares filled so far
    price:        float         # limit price of the order
    side:         str           # "BUY" or "SELL"
    token_id:     str
    received_at:  float = field(default_factory=time.monotonic)


class FillCache:
    """
    Thread-safe in-memory store of order states from the user WS channel.

    Designed for high-frequency reads (MarketMaker checks on every BookCache
    event) and low-frequency writes (fills arrive on user WS, typically
    seconds apart).

    Parameters
    ----------
    max_age_seconds : float
        Completed orders (FILLED/CANCELLED) are retained for this long,
        then purged to avoid unbounded memory growth.  Default: 300s (5min).
    """

    def __init__(self, max_age_seconds: float = 300.0) -> None:
        self._orders:  dict[str, OrderEvent] = {}
        self._lock     = asyncio.Lock()
        self._max_age  = max_age_seconds
        # Event fired after any fill/cancel — Brain/MM can await this
        self._updated: asyncio.Event = asyncio.Event()

    # ── Write path (called by UserWsClient) ───────────────────────────────────

    async def record_event(
        self,
        order_id:     str,
        status:       FillStatus,
        size_matched: float = 0.0,
        price:        float = 0.0,
        side:         str   = "",
        token_id:     str   = "",
    ) -> None:
        """Store or update an order's state from a WS user event."""
        event = OrderEvent(
            order_id=order_id,
            status=status,
            size_matched=size_matched,
            price=price,
            side=side,
            token_id=token_id,
        )
        async with self._lock:
            self._orders[order_id] = event
            self._purge_stale_locked()

        if status in (FillStatus.FILLED, FillStatus.CANCELLED):
            log.info(
                "FillCache: order %s → %s (size_matched=%.2f price=%.3f)",
                order_id[:16], status.value, size_matched, price,
            )
            self._updated.set()
            self._updated.clear()

    # ── Read path (called by MarketMaker) — non-blocking ─────────────────────

    def is_filled(self, order_id: str) -> bool:
        """Return True if the WS confirmed this order was fully filled."""
        ev = self._orders.get(order_id)
        return ev is not None and ev.status == FillStatus.FILLED

    def is_cancelled(self, order_id: str) -> bool:
        """Return True if the WS confirmed this order was cancelled."""
        ev = self._orders.get(order_id)
        return ev is not None and ev.status == FillStatus.CANCELLED

    def is_open(self, order_id: str) -> bool:
        """Return True if the WS says this order is still live."""
        ev = self._orders.get(order_id)
        return ev is not None and ev.status in (FillStatus.OPEN, FillStatus.PARTIAL)

    def get_fill_price(self, order_id: str) -> Optional[float]:
        """Return the matched price of a filled order, or None."""
        ev = self._orders.get(order_id)
        if ev and ev.status == FillStatus.FILLED:
            return ev.price
        return None

    def get_event(self, order_id: str) -> Optional[OrderEvent]:
        """Return the full OrderEvent for an order, or None if unknown."""
        return self._orders.get(order_id)

    def consume(self, order_id: str) -> Optional[OrderEvent]:
        """
        Remove and return the OrderEvent for an order_id.
        Called by MarketMaker after acting on a fill/cancel so the
        cache doesn't accumulate acknowledged events indefinitely.
        """
        return self._orders.pop(order_id, None)

    @property
    def updated_event(self) -> asyncio.Event:
        """
        asyncio.Event that fires after each FILLED or CANCELLED event.
        MarketMaker can await this for instant fill notification:

            await fill_cache.updated_event.wait()
        """
        return self._updated

    def __len__(self) -> int:
        return len(self._orders)

    def __repr__(self) -> str:
        filled    = sum(1 for e in self._orders.values() if e.status == FillStatus.FILLED)
        cancelled = sum(1 for e in self._orders.values() if e.status == FillStatus.CANCELLED)
        open_n    = sum(1 for e in self._orders.values() if e.status in (FillStatus.OPEN, FillStatus.PARTIAL))
        return f"FillCache(open={open_n} filled={filled} cancelled={cancelled})"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _purge_stale_locked(self) -> None:
        """Remove completed orders older than max_age_seconds (call inside lock)."""
        cutoff = time.monotonic() - self._max_age
        to_remove = [
            oid for oid, ev in self._orders.items()
            if ev.status in (FillStatus.FILLED, FillStatus.CANCELLED)
            and ev.received_at < cutoff
        ]
        for oid in to_remove:
            del self._orders[oid]
        if to_remove:
            log.debug("FillCache: purged %d stale completed orders", len(to_remove))
