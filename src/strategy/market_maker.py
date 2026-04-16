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
from typing import Optional

from src.core.config import Settings
from src.core.database import Database
from src.polymarket.book_cache import BookCache
from src.polymarket.clob_client import AsyncClobClient
from src.polymarket.fill_cache import FillCache
from src.polymarket.models import Market, OrderBook, OrderRequest, OrderType, Side
from src.strategy.microstructure import MicrostructureAnalyzer
from src.strategy.regime_filter import RegimeFilter

log = logging.getLogger(__name__)

# Polymarket maker rebate estimate (25% of taker fee which is ~2%)
_REBATE_RATE = 0.005
_DEFENSIVE_OBI_THRESHOLD = -0.15
_DEFENSIVE_MAX_EFFECTIVE_SPREAD = 0.06


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
    bought_at: float = 0.0  # time.time() when BUY fill was confirmed
    state: str = "IDLE"   # IDLE | BUY_POSTED | BOUGHT | SELL_POSTED


class MarketMaker:
    """
    Spread-capture market maker — Trigger layer of the Brain/Trigger architecture.

    Consumes ExecutionSignal / CancelSignal from Brain via asyncio.Queue.
    Manages GTC order lifecycle: buy → fill detection → sell → profit log.

    Fill detection is fully event-driven:
    - In LIVE mode: UserWsClient pushes fill events into FillCache.
      `is_filled(order_id)` is a non-blocking dict lookup — no HTTP.
    - In DRY-RUN mode: fill is simulated against the BookCache best prices.
    - Fallback: if FillCache has no data for an order (e.g., during WS reconnect
      window), falls back to get_open_orders() HTTP as a safety net.
    """

    def __init__(
        self,
        clob:       AsyncClobClient,
        db:         Database,
        settings:   Settings,
        fill_cache: Optional[FillCache]  = None,
        book_cache: Optional[BookCache]  = None,
    ) -> None:
        self._clob       = clob
        self._db         = db
        self._cfg        = settings
        self._fill_cache = fill_cache   # None → HTTP fallback always
        self._book_cache = book_cache   # None → always HTTP for order book
        self._regime = RegimeFilter(
            max_consecutive_losses=settings.max_consecutive_losses,
        )
        obi_threshold = max(settings.obi_block_threshold, _DEFENSIVE_OBI_THRESHOLD)
        max_effective_spread = min(settings.max_effective_spread, _DEFENSIVE_MAX_EFFECTIVE_SPREAD)
        self._obi_threshold = obi_threshold
        self._micro = MicrostructureAnalyzer(
            obi_threshold=obi_threshold,
            max_spread=max_effective_spread,
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
        """Check regime + microstructure, fetch book, post a passive BUY at best_bid."""
        market = slot.market

        quality_ok, quality_reason, quality_detail = await self._passes_market_quality_filters(market)
        if not quality_ok:
            await self._record_rejection(slot, quality_reason or "quality", quality_detail)
            return {}

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
            await self._record_rejection(slot, "regime", verdict.reason)
            log.debug("MM regime blocked %s: %s", market.question[:40], verdict.reason)
            return {}

        # -- Order book depth check --
        book = await self._clob.get_order_book(slot.token_id)
        if book is None or book.best_bid is None:
            await self._record_rejection(slot, "book_missing", "book unavailable or best_bid missing")
            return {}

        bid_depth = book.depth_usd(Side.BUY, levels=5)
        if bid_depth < self._cfg.min_book_depth_usd:
            await self._record_rejection(
                slot,
                "depth",
                f"bid_depth={bid_depth:.2f} < min={self._cfg.min_book_depth_usd:.2f}",
            )
            log.debug("MM depth too low for %s: $%.0f", market.question[:40], bid_depth)
            return {}

        # -- Microstructure defense --
        toxicity = await self._db.get_toxicity_ratio(
            market.condition_id, self._cfg.toxicity_lookback,
        )
        signal = self._micro.analyze(
            book=book,
            spread_target=self._cfg.spread_target,
            toxicity_ratio=toxicity,
            toxicity_threshold=self._cfg.mm_toxicity_threshold,
        )
        if not signal.is_safe:
            await self._record_rejection(slot, self._micro_reason_code(signal.block_reason), signal.block_reason)
            log.info("MM micro blocked %s: %s", market.question[:40], signal.block_reason)
            return {}

        # -- Calculate order params --
        buy_price = book.best_bid
        if buy_price <= 0.01 or buy_price >= 0.99:
            await self._record_rejection(slot, "price_extreme", f"buy_price={buy_price:.3f}")
            return {}
        if buy_price < self._cfg.mm_min_entry_price or buy_price > self._cfg.mm_max_entry_price:
            await self._record_rejection(
                slot,
                "price_band",
                f"buy={buy_price:.3f} outside [{self._cfg.mm_min_entry_price:.3f}, {self._cfg.mm_max_entry_price:.3f}]",
            )
            log.debug(
                "MM entry blocked by price band for %s: buy=%.3f outside [%.3f, %.3f]",
                market.question[:40], buy_price, self._cfg.mm_min_entry_price, self._cfg.mm_max_entry_price,
            )
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
        """Check if the BUY order has been filled, gone stale, or OBI turned toxic."""
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
        book: OrderBook | None = None

        if self._cfg.dry_run:
            # Dry-run: simulate fill from BookCache (if available) or CLOB HTTP
            book = self._book_cache.get_order_book(slot.token_id) if self._book_cache else None
            if book is None:
                book = await self._clob.get_order_book(slot.token_id)
            if book and book.best_ask is not None and book.best_ask <= slot.buy_price:
                filled = True

        elif self._fill_cache is not None and slot.buy_order_id:
            # LIVE mode: check FillCache first (non-blocking, zero HTTP)
            if self._fill_cache.is_filled(slot.buy_order_id):
                filled = True
                self._fill_cache.consume(slot.buy_order_id)  # acknowledge
            elif self._fill_cache.is_cancelled(slot.buy_order_id):
                # Order was cancelled externally (unusual) — reset
                self._fill_cache.consume(slot.buy_order_id)
                log.warning(
                    "[MM] BUY externally cancelled: '%s' — resetting slot",
                    slot.market.question[:40],
                )
                self._reset_slot(slot)
                return {}
            # else: still OPEN — no action

        else:
            # Fallback: HTTP polling (no FillCache, or no order_id)
            open_orders = await self._clob.get_open_orders()
            order_ids = {o.get("id") or o.get("orderID") for o in open_orders}
            if slot.buy_order_id and slot.buy_order_id not in order_ids:
                filled = True

        # -- Aggressive OBI cancel: if sell pressure appeared, pull the order --
        if not filled and book is None:
            # Need fresh book for OBI check — use cache if available
            book = (
                self._book_cache.get_order_book(slot.token_id)
                if self._book_cache else None
            ) or await self._clob.get_order_book(slot.token_id)
        if not filled and book is not None:
            obi = self._micro.order_book_imbalance(book)
            if obi < self._obi_threshold:
                log.info(
                    "[MM] OBI cancel: '%s' OBI=%.2f — pulling buy order",
                    slot.market.question[:40], obi,
                )
                if not self._cfg.dry_run and slot.buy_order_id:
                    await self._clob.cancel_order(slot.buy_order_id)
                if slot.round_id:
                    await self._db.cancel_mm_round(slot.round_id)
                self._reset_slot(slot)
                return {}

        if filled:
            log.info(
                "[MM] BUY FILLED: %.0f shares '%s' @ %.3f",
                slot.shares, slot.market.question[:40], slot.buy_price,
            )
            if slot.round_id:
                await self._db.update_mm_round_bought(slot.round_id)
                # Track fill for toxicity analysis
                mid = book.mid_price if book else None
                await self._db.insert_mm_fill(
                    round_id=slot.round_id,
                    market_id=slot.market.condition_id,
                    side="BUY",
                    fill_price=slot.buy_price,
                    mid_price_after=mid,
                )
            slot.bought_at = time.time()
            slot.state = "BOUGHT"
            return await self._handle_bought(slot)

        return {}

    async def _handle_bought(self, slot: _MarketSlot) -> dict[str, int]:
        """Post a SELL order at entry + spread_target."""
        if slot.bought_at <= 0:
            slot.bought_at = time.time()
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
        """Check if the SELL order has been filled; also check stop-loss and trailing exit."""
        filled = False
        stop_hit = False
        timed_out = False

        # Get current book — use cache if available, else HTTP
        book = (
            self._book_cache.get_order_book(slot.token_id)
            if self._book_cache else None
        ) or await self._clob.get_order_book(slot.token_id)

        if self._cfg.dry_run:
            if book and book.best_bid is not None and book.best_bid >= slot.sell_price:
                filled = True
            elif book and book.mid_price is not None:
                sl_price = slot.buy_price * (1 - self._cfg.mm_stop_loss_pct)
                if book.mid_price <= sl_price:
                    stop_hit = True

        elif self._fill_cache is not None and slot.sell_order_id:
            # LIVE mode: event-driven fill detection — no HTTP
            if self._fill_cache.is_filled(slot.sell_order_id):
                filled = True
                self._fill_cache.consume(slot.sell_order_id)  # acknowledge
            elif self._fill_cache.is_cancelled(slot.sell_order_id):
                # Sell was cancelled externally — this is unexpected; repost
                self._fill_cache.consume(slot.sell_order_id)
                log.warning(
                    "[MM] SELL externally cancelled: '%s' @ %.3f — reposting",
                    slot.market.question[:40], slot.sell_price,
                )
                slot.state = "BOUGHT"
                return await self._handle_bought(slot)

            # Stop-loss check from BookCache (same logic, no extra HTTP)
            if not filled and book and book.mid_price is not None:
                sl_price = slot.buy_price * (1 - self._cfg.mm_stop_loss_pct)
                if book.mid_price <= sl_price:
                    stop_hit = True

        else:
            # Fallback: HTTP polling
            open_orders = await self._clob.get_open_orders()
            order_ids = {o.get("id") or o.get("orderID") for o in open_orders}
            if slot.sell_order_id and slot.sell_order_id not in order_ids:
                filled = True

        if not filled and slot.bought_at > 0:
            held_seconds = time.time() - slot.bought_at
            if held_seconds >= self._cfg.mm_max_hold_seconds:
                timed_out = True

        # -- Trailing exit: if price moved well above our sell, raise the sell --
        if not filled and not stop_hit and not timed_out and book and book.best_bid is not None:
            trail_threshold = slot.sell_price + self._cfg.spread_target
            if book.best_bid > trail_threshold:
                new_sell = min(book.best_bid, 0.99)
                log.info(
                    "[MM] TRAILING UP: '%s' old_sell=%.3f new_sell=%.3f (bid=%.3f)",
                    slot.market.question[:40], slot.sell_price, new_sell, book.best_bid,
                )
                if not self._cfg.dry_run and slot.sell_order_id:
                    await self._clob.cancel_order(slot.sell_order_id)
                    resp = await self._clob.place_order(OrderRequest(
                        token_id=slot.token_id,
                        price=round(new_sell, 2),
                        size=round(slot.shares, 2),
                        side=Side.SELL,
                        order_type=OrderType.GTC,
                    ))
                    slot.sell_order_id = resp.order_id
                else:
                    slot.sell_order_id = f"dry-sell-trail-{slot.market.condition_id[:8]}"

                slot.sell_price = new_sell
                if slot.round_id:
                    await self._db.update_mm_round_sell_posted(
                        slot.round_id, new_sell, slot.sell_order_id,
                    )

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
                mid = book.mid_price if book else None
                await self._db.insert_mm_fill(
                    round_id=slot.round_id,
                    market_id=slot.market.condition_id,
                    side="SELL",
                    fill_price=slot.sell_price,
                    mid_price_after=mid,
                )

            self._reset_slot(slot)
            return {"profits": 1}

        if stop_hit or timed_out:
            if book and book.best_bid is not None:
                exit_price = book.best_bid
            elif book and book.mid_price is not None:
                exit_price = book.mid_price
            else:
                exit_price = slot.buy_price * (1 - self._cfg.mm_stop_loss_pct)
            pnl = slot.shares * (exit_price - slot.buy_price)

            if stop_hit:
                log.info(
                    "[MM] STOP-LOSS: '%s' buy=%.3f exit=%.3f pnl=%+.4f",
                    slot.market.question[:40], slot.buy_price, exit_price, pnl,
                )
            else:
                held_minutes = (time.time() - slot.bought_at) / 60 if slot.bought_at > 0 else 0.0
                log.info(
                    "[MM] TIME-STOP: '%s' held=%.1fmin buy=%.3f exit=%.3f pnl=%+.4f",
                    slot.market.question[:40], held_minutes, slot.buy_price, exit_price, pnl,
                )

            if not self._cfg.dry_run and slot.sell_order_id:
                await self._clob.cancel_order(slot.sell_order_id)

            if slot.round_id:
                await self._db.close_mm_round(slot.round_id, exit_price, pnl, 0.0)

            self._reset_slot(slot)
            if pnl > 0:
                return {"profits": 1}
            if pnl < 0:
                return {"losses": 1}
            return {}

        return {}

    # ── Helpers ─────────────────────────────────────────────────────────────

    async def _passes_market_quality_filters(self, market: Market) -> tuple[bool, str | None, str]:
        if market.volume_24hr < self._cfg.mm_min_market_volume_24h_usd:
            log.debug(
                "MM market blocked (volume): %s vol24h=%.0f < %.0f",
                market.question[:40], market.volume_24hr, self._cfg.mm_min_market_volume_24h_usd,
            )
            return False, "volume", f"vol24h={market.volume_24hr:.0f} < min={self._cfg.mm_min_market_volume_24h_usd:.0f}"

        if market.liquidity < self._cfg.mm_min_market_liquidity_usd:
            log.debug(
                "MM market blocked (liquidity): %s liq=%.0f < %.0f",
                market.question[:40], market.liquidity, self._cfg.mm_min_market_liquidity_usd,
            )
            return False, "liquidity", f"liq={market.liquidity:.0f} < min={self._cfg.mm_min_market_liquidity_usd:.0f}"

        if not self._cfg.mm_blacklist_enabled:
            return True, None, ""

        stats = await self._db.get_mm_market_stats(
            market.condition_id,
            lookback=self._cfg.mm_blacklist_lookback_trades,
        )
        trades = int(stats.get("trades", 0))
        wins = int(stats.get("wins", 0))
        net_pnl = float(stats.get("net_pnl", 0.0))

        if trades < self._cfg.mm_blacklist_min_trades:
            return True, None, ""

        win_rate = (wins / trades) if trades else 0.0
        should_block = wins == 0 or (win_rate < self._cfg.mm_blacklist_min_win_rate and net_pnl <= 0)
        if should_block:
            log.info(
                "MM blacklist blocked %s: trades=%d wins=%d win_rate=%.1f%% net=%+.4f",
                market.question[:40], trades, wins, win_rate * 100.0, net_pnl,
            )
            detail = f"trades={trades} wins={wins} win_rate={win_rate:.2%} net_pnl={net_pnl:+.4f}"
            return False, "blacklist", detail
        return True, None, ""

    async def _record_rejection(self, slot: _MarketSlot, reason_code: str, detail: str = "") -> None:
        await self._db.insert_mm_rejection(
            market_id=slot.market.condition_id,
            token_id=slot.token_id,
            question=slot.market.question,
            reason_code=reason_code,
            detail=detail,
            slot_state=slot.state,
        )

    @staticmethod
    def _micro_reason_code(block_reason: str) -> str:
        text = (block_reason or "").lower()
        if "obi=" in text:
            return "obi"
        if "toxicity=" in text:
            return "toxicity"
        if "spread=" in text:
            return "spread"
        return "micro"

    def _reset_slot(self, slot: _MarketSlot) -> None:
        slot.state = "IDLE"
        slot.round_id = None
        slot.buy_order_id = None
        slot.sell_order_id = None
        slot.buy_price = 0.0
        slot.sell_price = 0.0
        slot.shares = 0.0
        slot.posted_at = 0.0
        slot.bought_at = 0.0

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
