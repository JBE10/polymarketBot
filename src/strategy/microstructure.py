"""
Order book microstructure analysis for defensive market making.

Provides indicators that detect when placing passive orders is dangerous
due to adverse selection (informed traders filling your stale quotes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.polymarket.models import OrderBook, Side

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicrostructureSignal:
    obi: float
    effective_spread: float
    is_safe: bool
    block_reason: str


class MicrostructureAnalyzer:
    """
    Lightweight order book analysis for market-making defense.

    Checks:
      - OBI (Order Book Imbalance): detects sell pressure
      - Effective spread: detects illiquidity or event-driven widening
      - Toxicity: checks historical fill adverse movement
    """

    def __init__(
        self,
        obi_threshold: float = -0.3,
        max_spread: float = 0.10,
        min_spread_for_entry: float = 0.0,
    ) -> None:
        self._obi_threshold = obi_threshold
        self._max_spread = max_spread
        self._min_spread = min_spread_for_entry

    def analyze(
        self,
        book: OrderBook,
        spread_target: float = 0.02,
        toxicity_ratio: float = 0.0,
        toxicity_threshold: float = 0.6,
    ) -> MicrostructureSignal:
        """
        Analyze the order book and return a safety signal.

        Args:
            book: Current order book snapshot
            spread_target: The MM's target spread (must be < book spread)
            toxicity_ratio: Fraction of recent fills that were toxic (0-1)
            toxicity_threshold: Block if toxicity exceeds this

        Returns:
            MicrostructureSignal with safety verdict and diagnostics
        """
        obi = self.order_book_imbalance(book)
        eff_spread = self.effective_spread(book)

        # --- OBI check: strong sell pressure ---
        if obi < self._obi_threshold:
            return MicrostructureSignal(
                obi=obi,
                effective_spread=eff_spread,
                is_safe=False,
                block_reason=f"OBI={obi:.2f} < {self._obi_threshold} (sell pressure)",
            )

        # --- Spread too wide: illiquid or event-driven ---
        if eff_spread > self._max_spread:
            return MicrostructureSignal(
                obi=obi,
                effective_spread=eff_spread,
                is_safe=False,
                block_reason=f"spread={eff_spread:.3f} > max {self._max_spread} (illiquid/event)",
            )

        # --- Spread too narrow: no room for profit ---
        if eff_spread < spread_target * 0.5:
            return MicrostructureSignal(
                obi=obi,
                effective_spread=eff_spread,
                is_safe=False,
                block_reason=f"spread={eff_spread:.3f} < target/2 (no margin)",
            )

        # --- Toxicity check ---
        if toxicity_ratio > toxicity_threshold:
            return MicrostructureSignal(
                obi=obi,
                effective_spread=eff_spread,
                is_safe=False,
                block_reason=f"toxicity={toxicity_ratio:.1%} > {toxicity_threshold:.0%} (adverse fills)",
            )

        return MicrostructureSignal(
            obi=obi,
            effective_spread=eff_spread,
            is_safe=True,
            block_reason="",
        )

    @staticmethod
    def order_book_imbalance(book: OrderBook, levels: int = 5) -> float:
        """
        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Range: -1.0 (all sell pressure) to +1.0 (all buy pressure).
        Values near 0 indicate balanced flow (good for MM).
        """
        bid_vol = sum(l.size for l in book.bids[:levels])
        ask_vol = sum(l.size for l in book.asks[:levels])
        total = bid_vol + ask_vol

        if total < 1e-9:
            return 0.0

        return (bid_vol - ask_vol) / total

    @staticmethod
    def effective_spread(book: OrderBook) -> float:
        """
        Volume-weighted effective spread across top 3 levels.

        More accurate than simple best_ask - best_bid because it accounts
        for thin top-of-book that hides the real cost of execution.
        """
        if not book.bids or not book.asks:
            return 1.0

        n = min(3, len(book.bids), len(book.asks))

        bid_sum = sum(book.bids[i].price * book.bids[i].size for i in range(n))
        bid_vol = sum(book.bids[i].size for i in range(n))

        ask_sum = sum(book.asks[i].price * book.asks[i].size for i in range(n))
        ask_vol = sum(book.asks[i].size for i in range(n))

        if bid_vol < 1e-9 or ask_vol < 1e-9:
            bb = book.best_bid
            ba = book.best_ask
            if bb is not None and ba is not None:
                return ba - bb
            return 1.0

        vwap_bid = bid_sum / bid_vol
        vwap_ask = ask_sum / ask_vol

        return vwap_ask - vwap_bid
