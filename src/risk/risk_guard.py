"""
Risk Guard — Portfolio-level and microstructure-level risk management.

Hierarchy of checks (fastest → most expensive)
-----------------------------------------------
1. OBI Emergency Kill (microsecond-class, runs in Brain)
   → cancel any open order when sell pressure becomes extreme

2. Volatility Circuit Breaker
   → cancel if effective spread tripled since order was posted

3. Time-sensitive Microstructure Guard
   → tighten OBI threshold in the last 3 minutes before resolution

4. Portfolio Guards (expensive, run before entry only)
   a. Daily drawdown kill switch
   b. Max total exposure cap
   c. Asset-level consecutive-loss cooldown

This separation ensures that the ultra-fast guards (1-3) never block
on a DB call, while the slower portfolio guards (4) run asynchronously
before placing a new order.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.polymarket.models import OrderBook

log = logging.getLogger(__name__)


class RiskGuard:
    def __init__(self, db, config) -> None:
        """
        Parameters
        ----------
        db     : Database instance
        config : RiskConfig / Settings derived from YAML
        """
        self.db     = db
        self.config = config

        # Portfolio guard thresholds
        self.max_total_exposure = getattr(config, "max_total_exposure_pct", 2.5) / 100.0
        self.max_daily_dd       = getattr(config, "max_daily_drawdown_pct", 1.5) / 100.0

        # Cooldown rules
        cooldown_cfg            = getattr(config, "consecutive_losses_cooldown", {})
        self.cooldown_enabled   = cooldown_cfg.get("enabled", True)
        self.cooldown_losses    = cooldown_cfg.get("losses", 2)

        # Microstructure thresholds
        self.obi_emergency_threshold: float = getattr(
            config, "obi_emergency_threshold", -0.30
        )
        # How much the spread can widen before we pull the order
        self.spread_volatility_multiplier: float = getattr(
            config, "spread_volatility_multiplier", 3.0
        )
        # Tighter OBI threshold in this many minutes before resolution
        self.pre_resolution_minutes: float = getattr(
            config, "pre_resolution_guard_minutes", 3.0
        )
        self.pre_resolution_obi: float = getattr(
            config, "pre_resolution_obi_threshold", -0.10
        )

    # ── 1. OBI Emergency Kill ─────────────────────────────────────────────────
    # NOTE: This is also called directly from Brain for sub-millisecond response.
    # RiskGuard.check_obi_emergency() is kept here as a synchronous helper so
    # it can be reused without awaiting.

    def check_obi_emergency(
        self,
        book: OrderBook,
        minutes_to_resolution: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Check if current OBI warrants an emergency order cancellation.

        Returns (should_cancel: bool, reason: str).
        This is synchronous and fast — no DB access.

        Parameters
        ----------
        book                   : current order book
        minutes_to_resolution  : if provided, applies tighter threshold near expiry
        """
        if not book.bids or not book.asks:
            return False, ""

        bid_vol = sum(l.size for l in book.bids[:5])
        ask_vol = sum(l.size for l in book.asks[:5])
        total   = bid_vol + ask_vol
        if total < 1e-9:
            return False, ""

        obi = (bid_vol - ask_vol) / total

        # Apply tighter threshold near resolution (last N minutes)
        if (
            minutes_to_resolution is not None
            and minutes_to_resolution <= self.pre_resolution_minutes
        ):
            threshold = self.pre_resolution_obi
            label     = f"pre_resolution({minutes_to_resolution:.1f}m)"
        else:
            threshold = self.obi_emergency_threshold
            label     = "normal"

        if obi < threshold:
            reason = (
                f"OBI emergency kill [{label}]: OBI={obi:.3f} < {threshold:.3f} "
                f"(bid_vol={bid_vol:.1f} ask_vol={ask_vol:.1f})"
            )
            return True, reason

        return False, ""

    # ── 2. Volatility Circuit Breaker ─────────────────────────────────────────

    def check_spread_explosion(
        self,
        current_book: OrderBook,
        spread_at_entry: float,
    ) -> tuple[bool, str]:
        """
        Detect if the spread has exploded since entry (event-driven volatility).

        If the effective spread is now more than `spread_volatility_multiplier`×
        the spread when the order was posted, pull the order.

        Returns (should_cancel: bool, reason: str).
        Synchronous — no DB access.
        """
        if (
            current_book.best_bid is None
            or current_book.best_ask is None
            or spread_at_entry <= 0
        ):
            return False, ""

        current_spread = current_book.best_ask - current_book.best_bid
        if current_spread <= 0:
            return False, ""

        ratio = current_spread / spread_at_entry
        if ratio >= self.spread_volatility_multiplier:
            reason = (
                f"Spread explosion: current={current_spread:.4f} "
                f"entry={spread_at_entry:.4f} "
                f"ratio={ratio:.1f}× >= {self.spread_volatility_multiplier:.1f}×"
            )
            return True, reason

        return False, ""

    # ── 3. Combined Microstructure Kill (convenience) ─────────────────────────

    def check_microstructure_kill(
        self,
        current_book: OrderBook,
        spread_at_entry: float = 0.0,
        minutes_to_resolution: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Run all synchronous microstructure checks.
        Returns (should_cancel, reason).
        """
        should_cancel, reason = self.check_obi_emergency(
            current_book, minutes_to_resolution
        )
        if should_cancel:
            return True, reason

        if spread_at_entry > 0:
            should_cancel, reason = self.check_spread_explosion(
                current_book, spread_at_entry
            )
            if should_cancel:
                return True, reason

        return False, ""

    # ── 4. Portfolio Guards (async — requires DB) ─────────────────────────────

    async def is_safe_to_trade(self, bankroll: float, asset_symbol: str) -> bool:
        """
        Run portfolio-level risk checks before entering a new trade.
        Returns True if all checks pass.

        Checks:
          a. Daily drawdown kill switch  (max_daily_drawdown_pct = 1.5%)
          b. Total exposure cap          (max_total_exposure_pct = 2.5%)
          c. Consecutive-loss cooldown   (per asset)
        """
        if bankroll <= 0:
            return False

        # a. Daily Drawdown — harder limit than before (1.5% not 3%)
        daily_pnl = await self.db.get_daily_mm_pnl()
        if daily_pnl < 0:
            dd_pct = abs(daily_pnl) / bankroll
            if dd_pct >= self.max_daily_dd:
                log.warning(
                    "RiskGuard BLOCKED: Daily Drawdown %.2f%% >= limit %.2f%%",
                    dd_pct * 100, self.max_daily_dd * 100,
                )
                return False

        # b. Total Exposure
        open_positions    = await self.db.get_open_positions()
        total_exposure    = sum(p.get("size_usd", 0) for p in open_positions)
        exposure_pct      = total_exposure / bankroll

        if exposure_pct >= self.max_total_exposure:
            log.warning(
                "RiskGuard BLOCKED: Exposure %.2f%% >= limit %.2f%%",
                exposure_pct * 100, self.max_total_exposure * 100,
            )
            return False

        # c. Consecutive Losses Cooldown
        if self.cooldown_enabled:
            consecutive = await self.db.get_consecutive_losses(asset_symbol)
            if consecutive >= self.cooldown_losses:
                log.warning(
                    "RiskGuard BLOCKED: Cooldown for %s (%d consecutive losses >= %d)",
                    asset_symbol, consecutive, self.cooldown_losses,
                )
                return False

        return True

    def validate_time_filter(
        self, minutes_to_end: Optional[float], min_required: float = 3.0
    ) -> bool:
        """Block trading in the volatile tail-end of a market's life."""
        if minutes_to_end is not None and minutes_to_end <= min_required:
            log.info(
                "RiskGuard BLOCKED: %.1f min to resolution < required %.1f min",
                minutes_to_end, min_required,
            )
            return False
        return True
