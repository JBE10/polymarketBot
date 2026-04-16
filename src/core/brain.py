"""
Brain — Event-Driven Analysis Engine.

Architecture
------------
The Brain sits between the WebSocket data feed and the execution engine:

    WsClient ──push──► BookCache ──event──► Brain ──signal──► ExecutionQueue
                                                                │
                                            Trigger (MarketMaker) ◄──────────

The Brain wakes up ONLY when the BookCache signals a new update (via its
asyncio.Event), so it never burns CPU in an idle spin loop.

Responsibilities
----------------
1.  For each active market slot:
    a.  Read the latest BookSnapshot from the cache (non-blocking).
    b.  Run OBI check (immediate kill if toxic).
    c.  Run MicrostructureAnalyzer (spread, toxicity).
    d.  Run RegimeFilter (trend, volatility).
    e.  Run cost model simulation (fee + slippage → net edge).
    f.  If all checks pass: publish an ExecutionSignal to the queue.

2.  Emergency OBI kill: if a market with an open position shows OBI below
    the emergency threshold, publish a CancelSignal BEFORE the Trigger
    can act — this is the closest we get to microsecond risk management
    in Python without Rust/Go.

3.  Update the subscription list for WsClient when approved markets change.

Signals
-------
    ExecutionSignal   — place a new buy limit order
    CancelSignal      — cancel an existing order NOW (OBI/volatility kill)
    RollSignal        — adjust sell price (trailing exit)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from src.polymarket.book_cache import BookCache, BookSnapshot
from src.polymarket.models import Market, Side
from src.strategy.microstructure import MicrostructureAnalyzer
from src.strategy.regime_filter import RegimeFilter
from src.strategy.cost_model import CostModel, simulate_market_impact

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.core.database import Database

log = logging.getLogger(__name__)

# ── Signal types ──────────────────────────────────────────────────────────────


class SignalKind(Enum):
    EXECUTE = auto()   # place a new order
    CANCEL  = auto()   # cancel an existing order (risk kill)
    ROLL    = auto()   # move an existing sell order higher


@dataclass
class ExecutionSignal:
    """
    Instruction for the Trigger to place a new limit buy.

    All analysis is pre-computed here so the Trigger is as thin as possible.
    """
    kind:           SignalKind
    market_id:      str
    token_id:       str
    side:           Side
    price:          float          # computed best entry price
    shares:         float          # Kelly-sized quantity
    obi_at_signal:  float          # OBI snapshot that triggered entry
    net_edge_usd:   float          # expected profit after all costs
    emitted_at:     float = field(default_factory=time.monotonic)


@dataclass
class CancelSignal:
    """Instruction for the Trigger to cancel an order immediately."""
    kind:         SignalKind = SignalKind.CANCEL
    market_id:    str        = ""
    order_id:     str        = ""
    reason:       str        = ""
    emitted_at:   float      = field(default_factory=time.monotonic)


@dataclass
class RollSignal:
    """Instruction to raise a sell order to the new best bid."""
    kind:         SignalKind = SignalKind.ROLL
    market_id:    str        = ""
    token_id:     str        = ""
    order_id:     str        = ""
    new_price:    float      = 0.0
    emitted_at:   float      = field(default_factory=time.monotonic)


# ── Brain ─────────────────────────────────────────────────────────────────────


class Brain:
    """
    Event-driven analysis engine.

    Parameters
    ----------
    book_cache       : shared BookCache populated by WsClient
    signal_queue     : asyncio.Queue that Trigger reads signals from
    db               : Database for toxicity, consecutive-loss lookups
    settings         : global Settings (thresholds, sizing, etc.)
    """

    def __init__(
        self,
        book_cache:   BookCache,
        signal_queue: asyncio.Queue,
        db,
        settings,
    ) -> None:
        self._cache   = book_cache
        self._queue   = signal_queue
        self._db      = db
        self._cfg     = settings

        # Analysis components
        self._micro   = MicrostructureAnalyzer(
            obi_threshold=getattr(settings, "obi_block_threshold", -0.15),
            max_spread=getattr(settings, "max_effective_spread", 0.06),
        )
        self._regime  = RegimeFilter(
            max_consecutive_losses=getattr(settings, "max_consecutive_losses", 3),
        )
        self._cost    = CostModel(
            taker_fee_rate=getattr(settings, "taker_fee_rate", 0.002),
            gas_gwei_budget=getattr(settings, "gas_gwei_budget", 100.0),
        )

        # Emergency OBI threshold — tighter than the entry threshold
        self._obi_emergency: float = getattr(
            settings, "obi_emergency_threshold", -0.30
        )

        # Active market slots (managed by update_approved_markets)
        self._approved: dict[str, Market] = {}     # condition_id → Market
        # Open positions tracked by the Trigger (fed back via update_open_slots)
        self._open_slots: dict[str, dict] = {}     # condition_id → slot info

        self._stop_event: asyncio.Event = asyncio.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main event loop.

        Wakes on every BookCache update instead of on a fixed timer.
        This means latency is bounded by (WS message → BookCache.update → Brain
        check → signal enqueue) — typically < 1ms in pure Python with uvloop.
        """
        log.info("Brain: event loop started (obi_emergency=%.2f)", self._obi_emergency)

        while not self._stop_event.is_set():
            # Wait for the next book update (non-blocking for the event loop)
            try:
                await asyncio.wait_for(
                    self._cache.updated_event.wait(),
                    timeout=1.0,   # fallback poll in case we miss a signal
                )
            except asyncio.TimeoutError:
                pass  # 1s timeout: run checks anyway to catch stale state

            if self._stop_event.is_set():
                break

            await self._evaluate_all()

    async def stop(self) -> None:
        self._stop_event.set()

    # ── Market management ─────────────────────────────────────────────────────

    def update_approved_markets(self, markets: dict[str, Market]) -> None:
        """Called by the slow LLM loop to refresh the approved market set."""
        self._approved = dict(markets)
        log.debug("Brain: approved markets updated — %d markets", len(markets))

    def update_open_slots(self, slots: dict[str, dict]) -> None:
        """
        Called by the Trigger to inform the Brain of open positions.
        Used for emergency OBI kill decisions.
        Format: {condition_id: {"order_id": str, "state": str, "buy_price": float}}
        """
        self._open_slots = dict(slots)

    # ── Evaluation loop ───────────────────────────────────────────────────────

    async def _evaluate_all(self) -> None:
        """Evaluate all approved markets on the latest book data."""
        for mid, market in list(self._approved.items()):
            yes_token = market.yes_token
            if not yes_token:
                continue

            snap = self._cache.get(yes_token.token_id)
            if snap is None:
                continue  # no live data yet — skip

            # ── Emergency OBI kill (highest priority) ──────────────────────
            slot = self._open_slots.get(mid)
            if slot and slot.get("state") in ("BUY_POSTED", "SELL_POSTED"):
                await self._check_emergency_kill(market, snap, slot)
                continue  # don't evaluate entry while position is open

            # ── Entry evaluation ───────────────────────────────────────────
            if mid not in self._open_slots:
                await self._evaluate_entry(market, snap)

    async def _check_emergency_kill(
        self, market: Market, snap: BookSnapshot, slot: dict
    ) -> None:
        """
        OBI-triggered emergency cancel.

        If sell pressure appears while we have an open order, we cancel
        immediately WITHOUT waiting for the Trigger's next tick.
        This is the microstructure kill switch described in the architecture.
        """
        book = snap.to_order_book()
        obi  = self._micro.order_book_imbalance(book)

        if obi >= self._obi_emergency:
            return  # OBI is fine — no kill needed

        order_id = slot.get("order_id", "")
        reason   = f"OBI={obi:.3f} < emergency={self._obi_emergency:.3f}"

        log.warning(
            "Brain EMERGENCY KILL: %s order_id=%s %s",
            market.question[:50], order_id[:12], reason,
        )

        cancel = CancelSignal(
            market_id=market.condition_id,
            order_id=order_id,
            reason=reason,
        )
        try:
            self._queue.put_nowait(cancel)
        except asyncio.QueueFull:
            log.error("Brain: signal queue full — EMERGENCY CANCEL DROPPED for %s", market.condition_id[:12])

    async def _evaluate_entry(self, market: Market, snap: BookSnapshot) -> None:
        """Run all entry checks and emit an ExecutionSignal if warranted."""
        book = snap.to_order_book()

        # 1. Microstructure check
        toxicity = await self._db.get_toxicity_ratio(
            market.condition_id,
            getattr(self._cfg, "toxicity_lookback", 20),
        )
        signal = self._micro.analyze(
            book=book,
            spread_target=getattr(self._cfg, "spread_target", 0.02),
            toxicity_ratio=toxicity,
            toxicity_threshold=getattr(self._cfg, "mm_toxicity_threshold", 0.6),
        )
        if not signal.is_safe:
            log.debug("Brain: micro blocked %s — %s", market.question[:40], signal.block_reason)
            return

        # 2. Regime filter
        losses = await self._db.get_consecutive_losses(market.condition_id)
        verdict = self._regime.is_safe(
            prices=[],           # regime filter uses toxicity + losses for now
            volume_24h=market.volume_24hr,
            consecutive_losses=losses,
        )
        if not verdict:
            log.debug("Brain: regime blocked %s — %s", market.question[:40], verdict.reason)
            return

        # 3. Price band check
        buy_price = snap.best_bid
        if buy_price is None:
            return
        min_entry = getattr(self._cfg, "mm_min_entry_price", 0.05)
        max_entry = getattr(self._cfg, "mm_max_entry_price", 0.95)
        if not (min_entry <= buy_price <= max_entry):
            return

        # 4. Cost model — would we make money after fees + slippage?
        order_size_usd = getattr(self._cfg, "mm_order_size_usd", 10.0)
        shares         = order_size_usd / buy_price

        cost = simulate_market_impact(
            book=book,
            side=Side.BUY,
            size_shares=shares,
            cost_model=self._cost,
        )
        if cost.net_edge_usd <= 0:
            log.debug(
                "Brain: no edge after costs for %s — net=%.4f",
                market.question[:40], cost.net_edge_usd,
            )
            return

        # 5. All checks passed → emit signal
        ex_signal = ExecutionSignal(
            kind=SignalKind.EXECUTE,
            market_id=market.condition_id,
            token_id=market.yes_token.token_id,
            side=Side.BUY,
            price=round(buy_price, 3),
            shares=round(shares, 2),
            obi_at_signal=signal.obi,
            net_edge_usd=cost.net_edge_usd,
        )

        try:
            self._queue.put_nowait(ex_signal)
            log.info(
                "Brain: EXECUTE signal %s @ %.3f (%.0f shares) OBI=%.2f net=+%.4f",
                market.question[:40], buy_price, shares, signal.obi, cost.net_edge_usd,
            )
        except asyncio.QueueFull:
            log.warning("Brain: signal queue full — execution signal dropped")
