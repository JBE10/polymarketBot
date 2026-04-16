"""
Execution Cost Model — Fee & Slippage Simulation.

Why this matters
----------------
Naive market makers assume they'll get the best bid/ask price.
In reality:
  - If your order is larger than the top-of-book liquidity, you "sweep"
    multiple price levels — each level is worse than the last.
  - The platform charges a taker fee on every fill.
  - On Polygon, there's a gas fee per on-chain state change.

This module simulates the exact cost of filling `size_shares` worth of
an order by walking the order book level-by-level, then adding fees.

Usage
-----
    model = CostModel(taker_fee_rate=0.002, gas_gwei_budget=100.0)

    cost = simulate_market_impact(
        book=order_book_snapshot,
        side=Side.BUY,
        size_shares=100.0,
        cost_model=model,
    )

    if cost.net_edge_usd > 0:
        place_order(...)
    else:
        log("No edge after costs — skip")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.polymarket.models import OrderBook, PriceLevel, Side

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Polygon MATIC/USD approximate (updated periodically via config)
_DEFAULT_MATIC_USD = 0.80

# Approximate gas units per Polymarket CTF order on Polygon
_GAS_UNITS_PER_ORDER = 120_000


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CostModel:
    """
    Configurable cost parameters.

    Attributes
    ----------
    taker_fee_rate   : Platform taker fee as a decimal (0.002 = 0.2%)
    maker_fee_rate   : Platform maker fee (usually 0 on Polymarket)
    gas_gwei_budget  : Gas price in Gwei to use for Polygon cost estimate
    matic_price_usd  : MATIC/USD price for gas → USD conversion
    extra_slippage_buffer : Additional buffer applied to slippage (accounts
                            for latency between analysis and execution)
    """
    taker_fee_rate:          float = 0.002
    maker_fee_rate:          float = 0.000
    gas_gwei_budget:         float = 50.0
    matic_price_usd:         float = _DEFAULT_MATIC_USD
    extra_slippage_buffer:   float = 0.001   # 0.1% latency buffer


@dataclass(frozen=True)
class ExecutionCost:
    """
    Result of a market impact simulation.

    Attributes
    ----------
    best_price_available  : top-of-book price at analysis time
    avg_fill_price        : volume-weighted average fill price across all levels
    swept_levels          : number of price levels consumed
    fill_ratio            : fraction of requested size actually fillable
                            (< 1.0 means the book is too thin)
    slippage_usd          : cost from price movement vs best price
    platform_fee_usd      : exchange taker fee in USD
    gas_fee_usd           : estimated Polygon gas cost in USD
    total_cost_usd        : slippage + platform_fee + gas_fee + buffer
    spread_captured_usd   : estimated maker rebate / spread profit
    net_edge_usd          : spread_captured - total_cost
                            POSITIVE = trade is worth executing
    """
    best_price_available:  float
    avg_fill_price:        float
    swept_levels:          int
    fill_ratio:            float      # 0.0 – 1.0
    slippage_usd:          float
    platform_fee_usd:      float
    gas_fee_usd:           float
    total_cost_usd:        float
    spread_captured_usd:   float
    net_edge_usd:          float

    @property
    def is_profitable(self) -> bool:
        return self.net_edge_usd > 0

    def __str__(self) -> str:
        return (
            f"ExecutionCost("
            f"avg_fill={self.avg_fill_price:.4f} "
            f"levels={self.swept_levels} "
            f"fill_ratio={self.fill_ratio:.1%} "
            f"slip=${self.slippage_usd:.4f} "
            f"fee=${self.platform_fee_usd:.4f} "
            f"gas=${self.gas_fee_usd:.4f} "
            f"net={'+'if self.net_edge_usd>=0 else ''}{self.net_edge_usd:.4f}"
            f")"
        )


# ── Core simulation function ──────────────────────────────────────────────────


def simulate_market_impact(
    book:         OrderBook,
    side:         Side,
    size_shares:  float,
    cost_model:   CostModel = CostModel(),
    spread_target: float = 0.02,   # the MM's target spread for rebate estimate
) -> ExecutionCost:
    """
    Simulate the cost of filling `size_shares` by sweeping the order book.

    For a BUY order: we walk the ask side of the book from lowest to highest.
    For a SELL order: we walk the bid side from highest to lowest.

    This gives the true economic cost before any position is placed —
    enabling the Brain to reject trades that look profitable naively but
    actually lose money after all costs.

    Parameters
    ----------
    book          : current order book snapshot (asks/bids populated)
    side          : BUY or SELL
    size_shares   : number of shares to fill
    cost_model    : fee + gas parameters
    spread_target : expected spread capture for maker rebate

    Returns
    -------
    ExecutionCost with net_edge_usd > 0 meaning the trade is worth doing.
    """
    if size_shares <= 0:
        return _zero_cost(book, side)

    # Choose the relevant side of the book
    levels: list[PriceLevel] = book.asks if side == Side.BUY else book.bids
    if not levels:
        # No liquidity at all — cost is massive, return negative edge
        return _no_liquidity_cost(book, side, size_shares, cost_model)

    best_price = levels[0].price

    # Walk levels accumulating fill
    remaining    = size_shares
    total_value  = 0.0
    swept        = 0

    for level in levels:
        if remaining <= 0:
            break
        taken      = min(remaining, level.size)
        total_value += taken * level.price
        remaining  -= taken
        swept      += 1

    filled_shares = size_shares - remaining
    fill_ratio    = filled_shares / size_shares if size_shares > 0 else 0.0

    if filled_shares <= 0:
        return _no_liquidity_cost(book, side, size_shares, cost_model)

    avg_fill = total_value / filled_shares

    # ── Cost components ────────────────────────────────────────────────────

    # Slippage: avg_fill vs best price, plus latency buffer
    if side == Side.BUY:
        raw_slippage = avg_fill - best_price
    else:
        raw_slippage = best_price - avg_fill

    slippage_usd = max(0.0, raw_slippage + cost_model.extra_slippage_buffer) * filled_shares

    # Platform taker fee on notional value
    notional          = avg_fill * filled_shares
    platform_fee_usd  = notional * cost_model.taker_fee_rate

    # Gas cost (Polygon): gwei * gas_units → GWEI → MATIC → USD
    gas_fee_usd = (
        cost_model.gas_gwei_budget
        * 1e-9           # gwei → MATIC
        * _GAS_UNITS_PER_ORDER
        * cost_model.matic_price_usd
    )

    total_cost_usd = slippage_usd + platform_fee_usd + gas_fee_usd

    # ── Spread captured (maker side rebate estimate) ───────────────────────
    # As a passive market maker, we expect to capture the spread_target
    # on the filled quantity.  This is the gross P&L before costs.
    spread_captured_usd = spread_target * filled_shares

    net_edge_usd = spread_captured_usd - total_cost_usd

    return ExecutionCost(
        best_price_available=best_price,
        avg_fill_price=avg_fill,
        swept_levels=swept,
        fill_ratio=fill_ratio,
        slippage_usd=slippage_usd,
        platform_fee_usd=platform_fee_usd,
        gas_fee_usd=gas_fee_usd,
        total_cost_usd=total_cost_usd,
        spread_captured_usd=spread_captured_usd,
        net_edge_usd=net_edge_usd,
    )


def min_spread_for_profitability(
    size_shares:  float,
    entry_price:  float,
    cost_model:   CostModel = CostModel(),
) -> float:
    """
    Return the minimum spread (in price units) needed to break even after
    platform fees and gas.

    Useful for dynamically setting the spread_target rather than using a
    fixed value from config.
    """
    notional         = entry_price * size_shares
    fee              = notional * cost_model.taker_fee_rate
    gas              = (
        cost_model.gas_gwei_budget * 1e-9 * _GAS_UNITS_PER_ORDER * cost_model.matic_price_usd
    )
    buffer_cost      = cost_model.extra_slippage_buffer * size_shares
    total_fixed_cost = fee + gas + buffer_cost

    # spread_target * size_shares >= total_fixed_cost
    return total_fixed_cost / size_shares if size_shares > 0 else 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────


def _zero_cost(book: OrderBook, side: Side) -> ExecutionCost:
    best = (book.best_ask if side == Side.BUY else book.best_bid) or 0.0
    return ExecutionCost(
        best_price_available=best,
        avg_fill_price=best,
        swept_levels=0,
        fill_ratio=0.0,
        slippage_usd=0.0,
        platform_fee_usd=0.0,
        gas_fee_usd=0.0,
        total_cost_usd=0.0,
        spread_captured_usd=0.0,
        net_edge_usd=0.0,
    )


def _no_liquidity_cost(
    book: OrderBook,
    side: Side,
    size_shares: float,
    cost_model: CostModel,
) -> ExecutionCost:
    best = (book.best_ask if side == Side.BUY else book.best_bid) or 0.0
    gas  = (
        cost_model.gas_gwei_budget * 1e-9 * _GAS_UNITS_PER_ORDER * cost_model.matic_price_usd
    )
    return ExecutionCost(
        best_price_available=best,
        avg_fill_price=best,
        swept_levels=0,
        fill_ratio=0.0,
        slippage_usd=size_shares * best,   # effectively all slippage
        platform_fee_usd=0.0,
        gas_fee_usd=gas,
        total_cost_usd=size_shares * best + gas,
        spread_captured_usd=0.0,
        net_edge_usd=-(size_shares * best + gas),
    )
