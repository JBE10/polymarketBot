"""
Spread-capture market maker for Polymarket.

Runs on a fast loop (every mm_cycle_seconds) and manages passive GTC
limit orders to capture micro-profits from price oscillations.

Lifecycle per market:
    1. Post a GTC limit BUY at best_bid (passive — earns maker rebate)
    2. When BUY fills → immediately post a GTC limit SELL at entry + spread_target
    3. When SELL fills → log profit, reposition a new BUY
    4. Circuit breakers can halt a market at any stage

In dry-run mode, fills are simulated: a BUY is considered filled if the
current ask drops to or below the buy price, and a SELL fills if the
current bid rises to or above the sell price.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.core.config import Settings
from src.core.database import Database
from src.polymarket.clob_client import AsyncClobClient
from src.polymarket.models import Market, OrderRequest, OrderType, Side
from src.strategy.regime_filter import RegimeFilter

log = logging.getLogger(__name__)

# Polymarket maker rebate estimate (25% of taker fee which is ~2%)
_REBATE_RATE = 0.005


@dataclass
class _MarketSlot:
    """In-memory state for one market being made."""
    market: Market
    token_id: str
    round_id: int | None = None
    buy_order_id: str | None = None
    sell_order_id: str | None = None
    buy_price: float = 0.0
    sell_price: float = 0.0
    shares: float = 0.0
    posted_at: float = 0.0  # time.time() when BUY was posted
    state: str = "IDLE"   # IDLE | BUY_POSTED | BOUGHT | SELL_POSTED


class MarketMaker:
    """
    Manages spread-capture orders across multiple markets concurrently.

    The LLM evaluator feeds approved markets into `update_active_markets()`.
    Each fast-loop tick, `run_tick()` advances the state machine for every slot.
    """

    def __init__(
        self,
        clob: AsyncClobClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._clob = clob
        self._db = db
        self._cfg = settings
        self._regime = RegimeFilter(
            max_consecutive_losses=settings.max_consecutive_losses,
        )
        self._slots: dict[str, _MarketSlot] = {}
        self._approved_markets: dict[str, Market] = {}

    # ── Public interface ────────────────────────────────────────────────────

    def update_active_markets(self, markets: dict[str, Market]) -> None:
        """Called by the slow loop after LLM evaluation to refresh the market set."""
        self._approved_markets = dict(markets)

    async def run_tick(self) -> dict[str, int]:
        """
        One fast-loop tick.  Returns summary: {buys_posted, sells_posted, profits, losses}.
        """
        summary = {"buys_posted": 0, "sells_posted": 0, "profits": 0, "losses": 0}

        # Reconcile slots with approved markets
        self._reconcile_slots()

        for market_id, slot in list(self._slots.items()):
            try:
                result = await self._tick_slot(slot)
                for k, v in result.items():
                    summary[k] = summary.get(k, 0) + v
            except Exception as exc:
                log.error("MM tick error for %s: %s", market_id[:10], exc)

        return summary

    async def cancel_all(self) -> None:
        """Cancel all outstanding MM orders (called on shutdown)."""
        for slot in self._slots.values():
            await self._cancel_slot(slot)

    # ── Slot management ────────────────────────────────────────────────────

    def _reconcile_slots(self) -> None:
        """Add new approved markets, remove unapproved idle slots."""
        max_slots = self._cfg.max_mm_markets

        # Remove idle slots for markets no longer approved
        for mid in list(self._slots):
            if mid not in self._approved_markets and self._slots[mid].state == "IDLE":
                del self._slots[mid]

        # Add new approved markets up to the cap
        for mid, market in self._approved_markets.items():
            if mid in self._slots:
                self._slots[mid].market = market
                continue
            if len(self._slots) >= max_slots:
                break
            yes_token = market.yes_token
            if yes_token is None:
                continue
            self._slots[mid] = _MarketSlot(
                market=market,
                token_id=yes_token.token_id,
            )

    # ── Per-slot state machine ─────────────────────────────────────────────

    async def _tick_slot(self, slot: _MarketSlot) -> dict[str, int]:
        result: dict[str, int] = {}
        market = slot.market

        if slot.state == "IDLE":
            result = await self._handle_idle(slot)
        elif slot.state == "BUY_POSTED":
            result = await self._handle_buy_posted(slot)
        elif slot.state == "BOUGHT":
            result = await self._handle_bought(slot)
        elif slot.state == "SELL_POSTED":
            result = await self._handle_sell_posted(slot)

        return result

    async def _handle_idle(self, slot: _MarketSlot) -> dict[str, int]:
        """Check regime, fetch book, post a passive BUY at best_bid."""
        market = slot.market

        # -- Regime filter --
        losses = await self._db.get_consecutive_losses(market.condition_id)
        trades = await self._clob.get_trades(slot.token_id, limit=30)
        prices = [float(t.get("price", 0)) for t in trades if t.get("price")]

        verdict = self._regime.is_safe(
            prices=prices,
            volume_24h=market.volume_24hr,
            consecutive_losses=losses,
        )
        if not verdict:
            log.debug("MM regime blocked %s: %s", market.question[:40], verdict.reason)
            return {}

        # -- Order book depth check --
        book = await self._clob.get_order_book(slot.token_id)
        if book is None or book.best_bid is None:
            return {}

        bid_depth = book.depth_usd(Side.BUY, levels=5)
        if bid_depth < self._cfg.min_book_depth_usd:
            log.debug("MM depth too low for %s: $%.0f", market.question[:40], bid_depth)
            return {}

        # -- Calculate order params --
        buy_price = book.best_bid
        if buy_price <= 0.01 or buy_price >= 0.99:
            return {}

        shares = self._cfg.mm_order_size_usd / buy_price

        # -- Post BUY order --
        order_id: str | None = None
        if self._cfg.dry_run:
            order_id = f"dry-buy-{market.condition_id[:8]}"
            log.info(
                "[MM-DRY] POST BUY %.0f shares '%s' @ %.3f ($%.2f)",
                shares, market.question[:40], buy_price, self._cfg.mm_order_size_usd,
            )
        else:
            resp = await self._clob.place_order(OrderRequest(
                token_id=slot.token_id,
                price=round(buy_price, 2),
                size=round(shares, 2),
                side=Side.BUY,
                order_type=OrderType.GTC,
            ))
            if not resp.success and resp.order_id is None:
                log.warning("MM BUY order rejected for %s: %s", market.question[:40], resp.error_message)
                return {}
            order_id = resp.order_id

        round_id = await self._db.insert_mm_round(
            market_id=market.condition_id,
            token_id=slot.token_id,
            question=market.question,
            buy_price=buy_price,
            shares=shares,
            buy_order_id=order_id,
        )

        slot.state = "BUY_POSTED"
        slot.round_id = round_id
        slot.buy_order_id = order_id
        slot.buy_price = buy_price
        slot.shares = shares
        slot.posted_at = time.time()

        return {"buys_posted": 1}

    async def _handle_buy_posted(self, slot: _MarketSlot) -> dict[str, int]:
        """Check if the BUY order has been filled or has gone stale."""
        # -- Stale order cancellation --
        stale_secs = self._cfg.mm_stale_order_seconds
        elapsed = time.time() - slot.posted_at
        if slot.posted_at > 0 and elapsed > stale_secs:
            log.info(
                "[MM] BUY STALE after %ds: '%s' @ %.3f — cancelling",
                int(elapsed), slot.market.question[:40], slot.buy_price,
            )
            if not self._cfg.dry_run and slot.buy_order_id:
                await self._clob.cancel_order(slot.buy_order_id)
            if slot.round_id:
                await self._db.cancel_mm_round(slot.round_id)
            self._reset_slot(slot)
            return {}

        filled = False

        if self._cfg.dry_run:
            # Simulate fill: if current ask <= buy_price, consider it filled
            book = await self._clob.get_order_book(slot.token_id)
            if book and book.best_ask is not None and book.best_ask <= slot.buy_price:
                filled = True
        else:
            # Check if the order is still in the open orders list
            open_orders = await self._clob.get_open_orders()
            order_ids = {o.get("id") or o.get("orderID") for o in open_orders}
            if slot.buy_order_id and slot.buy_order_id not in order_ids:
                filled = True

        if filled:
            log.info(
                "[MM] BUY FILLED: %.0f shares '%s' @ %.3f",
                slot.shares, slot.market.question[:40], slot.buy_price,
            )
            if slot.round_id:
                await self._db.update_mm_round_bought(slot.round_id)
            slot.state = "BOUGHT"
            return await self._handle_bought(slot)

        return {}

    async def _handle_bought(self, slot: _MarketSlot) -> dict[str, int]:
        """Post a SELL order at entry + spread_target."""
        sell_price = min(slot.buy_price + self._cfg.spread_target, 0.99)

        order_id: str | None = None
        if self._cfg.dry_run:
            order_id = f"dry-sell-{slot.market.condition_id[:8]}"
            log.info(
                "[MM-DRY] POST SELL %.0f shares '%s' @ %.3f (target profit $%.2f)",
                slot.shares, slot.market.question[:40], sell_price,
                slot.shares * self._cfg.spread_target,
            )
        else:
            resp = await self._clob.place_order(OrderRequest(
                token_id=slot.token_id,
                price=round(sell_price, 2),
                size=round(slot.shares, 2),
                side=Side.SELL,
                order_type=OrderType.GTC,
            ))
            if not resp.success and resp.order_id is None:
                log.warning("MM SELL order rejected: %s", resp.error_message)
                return {}
            order_id = resp.order_id

        if slot.round_id:
            await self._db.update_mm_round_sell_posted(slot.round_id, sell_price, order_id)

        slot.state = "SELL_POSTED"
        slot.sell_order_id = order_id
        slot.sell_price = sell_price

        return {"sells_posted": 1}

    async def _handle_sell_posted(self, slot: _MarketSlot) -> dict[str, int]:
        """Check if the SELL order has been filled; also check stop-loss."""
        filled = False
        stop_hit = False

        book = await self._clob.get_order_book(slot.token_id)

        if self._cfg.dry_run:
            # Simulate sell fill: if current bid >= sell_price
            if book and book.best_bid is not None and book.best_bid >= slot.sell_price:
                filled = True
            # Simulate stop-loss: if mid_price dropped below entry - stop_loss_pct
            elif book and book.mid_price is not None:
                sl_price = slot.buy_price * (1 - self._cfg.stop_loss_pct)
                if book.mid_price <= sl_price:
                    stop_hit = True
        else:
            open_orders = await self._clob.get_open_orders()
            order_ids = {o.get("id") or o.get("orderID") for o in open_orders}
            if slot.sell_order_id and slot.sell_order_id not in order_ids:
                filled = True

        if filled:
            pnl = slot.shares * (slot.sell_price - slot.buy_price)
            rebate = slot.shares * slot.buy_price * _REBATE_RATE
            total = pnl + rebate

            log.info(
                "[MM] SELL FILLED: '%s' buy=%.3f sell=%.3f pnl=%+.4f rebate=+%.4f total=%+.4f",
                slot.market.question[:40], slot.buy_price, slot.sell_price, pnl, rebate, total,
            )

            if slot.round_id:
                await self._db.close_mm_round(
                    slot.round_id, slot.sell_price, pnl, rebate,
                )

            self._reset_slot(slot)
            return {"profits": 1}

        if stop_hit:
            exit_price = book.mid_price if book and book.mid_price else slot.buy_price * 0.9
            pnl = slot.shares * (exit_price - slot.buy_price)

            log.info(
                "[MM] STOP-LOSS: '%s' buy=%.3f exit=%.3f pnl=%+.4f",
                slot.market.question[:40], slot.buy_price, exit_price, pnl,
            )

            if not self._cfg.dry_run and slot.sell_order_id:
                await self._clob.cancel_order(slot.sell_order_id)

            if slot.round_id:
                await self._db.close_mm_round(slot.round_id, exit_price, pnl, 0.0)

            self._reset_slot(slot)
            return {"losses": 1}

        return {}

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _reset_slot(self, slot: _MarketSlot) -> None:
        slot.state = "IDLE"
        slot.round_id = None
        slot.buy_order_id = None
        slot.sell_order_id = None
        slot.buy_price = 0.0
        slot.sell_price = 0.0
        slot.shares = 0.0
        slot.posted_at = 0.0

    async def _cancel_slot(self, slot: _MarketSlot) -> None:
        """Cancel any outstanding orders and mark round as cancelled."""
        if not self._cfg.dry_run:
            if slot.buy_order_id and slot.state == "BUY_POSTED":
                await self._clob.cancel_order(slot.buy_order_id)
            if slot.sell_order_id and slot.state == "SELL_POSTED":
                await self._clob.cancel_order(slot.sell_order_id)
        if slot.round_id:
            await self._db.cancel_mm_round(slot.round_id)
        self._reset_slot(slot)
