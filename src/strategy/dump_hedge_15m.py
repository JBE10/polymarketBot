"""
dump_hedge_15m.py  —  Polymarket 15-min crypto pool arbitrage engine.

ESTRATEGIA
──────────
Polymarket publica mercados binarios de 15 minutos para BTC/ETH/SOL/XRP:
  "Will BTC be UP in the next 15 minutes?"
  → YES token  +  NO token  (siempre suman exactamente 1.00 USD al vencimiento)

Si el precio del mercado diverge de la eficiencia, aparecen dos oportunidades:

  1. SPREAD ARBITRAGE (sin riesgo direccional)
     ─────────────────────────────────────────
     Si  YES_ask + NO_ask < 1.00:
       • Compramos YES a Y  +  NO a N
       • Al vencimiento cobramos 1.00 sin importar el resultado
       • Ganancia por par: (1.00 - Y - N) × shares - fees
       Ejemplo: YES=0.31, NO=0.68 → suma=0.99 → EV=+0.01/share (1% libre de riesgo)

  2. DIRECTIONAL SHORT (momentum collapse)
     ───────────────────────────────────────
     Si YES_ask cae >12% desde el pico local en < 90s:
       • El mercado "dumpea" a YES (colapso de probabilidad)
       • Compramos NO (el complemento) que aún no refleja el dump
       • Target: vender NO cuando YES se recupera o mercado expira

MODO DE EJECUCIÓN
─────────────────
  DRY_RUN = true   → solo logs y DB events, sin órdenes reales
  DRY_RUN = false  → place_order() real vía CLOB L2 API

THROUGHPUT
──────────
  • Fetch paralelo: N mercados en 1 round-trip (asyncio.gather)
  • Ciclo de 2s: detecta oportunidades antes que el mercado corrija
  • HFT por volumen: muchas trades pequeñas, no una sola grande
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from src.polymarket.models import Market, OrderRequest, Side

log = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ArbOpportunity:
    """Represents a detected spread-arbitrage opportunity (YES + NO < 1.0)."""
    market_id:    str
    question:     str
    asset:        str
    yes_token_id: str
    no_token_id:  str
    yes_ask:      float
    no_ask:       float
    sum_price:    float          # yes_ask + no_ask
    edge:         float          # 1.0 - sum_price  (= guaranteed profit per $1)
    shares:       float          # position size in shares
    ts:           float = field(default_factory=time.time)

    @property
    def notional_usd(self) -> float:
        return self.shares * self.sum_price

    @property
    def expected_pnl(self) -> float:
        return self.shares * self.edge


@dataclass
class DirectionalState:
    """State for an active directional (momentum-collapse) position."""
    cycle_id:     str
    market_id:    str
    question:     str
    asset:        str
    leg1_side:    str           # "YES" | "NO" — the dumped side we faded
    leg1_price:   float         # price we entered the opposite leg at
    leg1_ts:      float
    status:       str = "LEG1_FILLED"


# ── Main engine ────────────────────────────────────────────────────────────────

class DumpHedge15MStrategy:
    """
    Dual-mode 15-min pool engine:
      • Mode A: SPREAD ARBITRAGE  — buy YES + NO simultaneously when sum < target
      • Mode B: DIRECTIONAL SHORT — fade a dumping leg (e.g. NO after BTC dump)

    Configuration (from .env / Settings):
      DH15M_ENABLED              = true
      DH15M_CYCLE_SECONDS        = 2
      DH15M_SUM_TARGET           = 0.95   # max YES+NO sum to trigger arb (5% edge)
      DH15M_MOVE_THRESHOLD       = 0.12   # 12% drop from peak to trigger directional
      DH15M_SHARES               = 25     # shares per leg
      DH15M_STOP_HEDGE_WAIT_SECONDS = 120 # max wait for leg2 (directional)
      DH15M_MIN_MINUTES_TO_END   = 5      # ignore markets expiring in <5 min
      DH15M_MAX_MINUTES_TO_END   = 20     # ignore markets with >20 min to end
    """

    def __init__(self, *, clob, db, settings) -> None:
        self._clob    = clob
        self._db      = db
        self._cfg     = settings
        # Price history for directional detection: key = "market_id:YES"|":NO"
        self._price_hist: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=600)
        )
        # Active directional positions waiting for leg 2
        self._active_dir: dict[str, DirectionalState] = {}
        # Track already-executed arb pairs to avoid double-entry in same window
        self._recent_arbs: dict[str, float] = {}   # market_id → timestamp

        # Stats for this session
        self._session_arbs       = 0
        self._session_dir_trades = 0
        self._session_pnl_est    = 0.0

    # ── Public interface ───────────────────────────────────────────────────────

    async def run_tick(self) -> dict[str, int]:
        """
        Single tick of the strategy loop.
        Returns summary dict for logging.
        """
        summary = {
            "candidates": 0,
            "arb_opportunities": 0,
            "arb_executed": 0,
            "dir_leg1":  0,
            "dir_hedged": 0,
            "dir_stopped": 0,
        }

        markets = await self._clob.get_markets(limit=self._cfg.dh15m_fetch_limit)
        now = time.time()

        # 1. Filter to only 15-min crypto pools
        candidates = [m for m in markets if self._is_candidate_market(m)]
        summary["candidates"] = len(candidates)

        if not candidates:
            return summary

        # Prune stale arb cooldown tracker (> 30s cooldown per market)
        self._recent_arbs = {
            mid: ts for mid, ts in self._recent_arbs.items() if now - ts < 30
        }

        # 2. Parallel fetch: all YES and NO books in one round-trip
        yes_coros = [
            self._clob.get_order_book(m.yes_token.token_id) if m.yes_token else _noop_book()
            for m in candidates
        ]
        no_coros = [
            self._clob.get_order_book(m.no_token.token_id) if m.no_token else _noop_book()
            for m in candidates
        ]

        yes_results, no_results = await asyncio.gather(
            asyncio.gather(*yes_coros, return_exceptions=True),
            asyncio.gather(*no_coros, return_exceptions=True),
        )

        # 3. Evaluate each market
        for m, yes_book, no_book in zip(candidates, yes_results, no_results):
            if not m.yes_token or not m.no_token:
                continue
            if isinstance(yes_book, Exception) or isinstance(no_book, Exception):
                continue
            if yes_book is None or no_book is None:
                continue
            if yes_book.best_ask is None or no_book.best_ask is None:
                continue

            yes_ask = float(yes_book.best_ask)
            no_ask  = float(no_book.best_ask)
            sum_p   = yes_ask + no_ask

            # Update price history for directional detection
            self._append_price(f"{m.condition_id}:YES", now, yes_ask)
            self._append_price(f"{m.condition_id}:NO",  now, no_ask)

            # ── MODE A: SPREAD ARBITRAGE ─────────────────────────────────────
            if sum_p <= self._cfg.dh15m_sum_target and m.condition_id not in self._recent_arbs:
                opp = ArbOpportunity(
                    market_id    = m.condition_id,
                    question     = m.question,
                    asset        = self._infer_asset(m.question),
                    yes_token_id = m.yes_token.token_id,
                    no_token_id  = m.no_token.token_id,
                    yes_ask      = yes_ask,
                    no_ask       = no_ask,
                    sum_price    = sum_p,
                    edge         = 1.0 - sum_p,
                    shares       = self._cfg.dh15m_shares,
                )
                summary["arb_opportunities"] += 1
                executed = await self._execute_arb(opp)
                if executed:
                    summary["arb_executed"] += 1
                    self._recent_arbs[m.condition_id] = now

            # ── MODE B: DIRECTIONAL (momentum collapse) ──────────────────────
            dir_state = self._active_dir.get(m.condition_id)

            if dir_state is None:
                leg1 = self._detect_leg1(m.condition_id, yes_ask=yes_ask, no_ask=no_ask)
                if leg1 is not None:
                    side, price = leg1
                    await self._open_directional(m, side, price, now)
                    summary["dir_leg1"] += 1
            else:
                result = await self._manage_directional(dir_state, yes_ask, no_ask, now)
                if result == "HEDGED":
                    summary["dir_hedged"] += 1
                elif result == "STOPPED":
                    summary["dir_stopped"] += 1

        return summary

    # ── Mode A: Spread arbitrage execution ────────────────────────────────────

    async def _execute_arb(self, opp: ArbOpportunity) -> bool:
        """
        Execute both legs of the arbitrage simultaneously.
        In dry-run: log + DB only.
        In live:    place_order() for both YES and NO legs.

        Returns True if the arb was submitted (dry or live).
        """
        fee_est = opp.notional_usd * 0.02   # ~2% Polymarket taker fee
        net_pnl = opp.expected_pnl - fee_est

        log.info(
            "[DH15M-ARB] %s | YES=%.3f + NO=%.3f = %.3f "
            "| edge=+%.4f | shares=%.0f | est_pnl=+$%.4f (net after fees)",
            opp.question[:55],
            opp.yes_ask, opp.no_ask, opp.sum_price,
            opp.edge, opp.shares, net_pnl,
        )

        if net_pnl <= 0:
            log.debug("[DH15M-ARB] SKIP: net_pnl negative after fees (edge %.4f < fee %.4f)", opp.edge, fee_est)
            return False

        # Log to DB
        cycle_id = f"arb-{opp.market_id[:12]}-{int(time.time())}"
        await self._db.insert_dump_hedge_event(
            cycle_id    = cycle_id,
            market_id   = opp.market_id,
            question    = opp.question,
            asset       = opp.asset,
            phase       = "ARB_BOTH_LEGS",
            leg1_side   = "YES",
            leg1_price  = opp.yes_ask,
            leg2_price  = opp.no_ask,
            sum_price   = opp.sum_price,
            shares      = opp.shares,
            status      = "DRY_RUN" if self._cfg.dry_run else "SUBMITTED",
            detail_json = json.dumps({
                "mode":     "spread_arb",
                "edge":     round(opp.edge, 6),
                "fee_est":  round(fee_est, 6),
                "net_pnl":  round(net_pnl, 6),
            }),
        )

        if self._cfg.dry_run:
            log.info(
                "[DH15M-ARB-DRY] BUY %.0f YES @ %.3f + %.0f NO @ %.3f | market='%s'",
                opp.shares, opp.yes_ask, opp.shares, opp.no_ask, opp.question[:45],
            )
            self._session_arbs      += 1
            self._session_pnl_est   += net_pnl
            return True

        # ── LIVE: place both orders in parallel ───────────────────────────────
        yes_req = OrderRequest(
            token_id = opp.yes_token_id,
            price    = round(opp.yes_ask, 2),
            size     = round(opp.shares, 2),
            side     = Side.BUY,
        )
        no_req = OrderRequest(
            token_id = opp.no_token_id,
            price    = round(opp.no_ask, 2),
            size     = round(opp.shares, 2),
            side     = Side.BUY,
        )

        yes_resp, no_resp = await asyncio.gather(
            self._clob.place_order(yes_req),
            self._clob.place_order(no_req),
            return_exceptions=True,
        )

        yes_ok = not isinstance(yes_resp, Exception) and getattr(yes_resp, "success", False)
        no_ok  = not isinstance(no_resp,  Exception) and getattr(no_resp,  "success", False)

        if yes_ok and no_ok:
            self._session_arbs    += 1
            self._session_pnl_est += net_pnl
            log.info(
                "[DH15M-ARB-LIVE] ✅ FILLED YES+NO | est P&L=+$%.4f | '%s'",
                net_pnl, opp.question[:45],
            )
        else:
            log.warning(
                "[DH15M-ARB-LIVE] ⚠️ partial fill: YES=%s NO=%s | '%s'",
                "OK" if yes_ok else "FAIL",
                "OK" if no_ok  else "FAIL",
                opp.question[:45],
            )

        return yes_ok and no_ok

    # ── Mode B: Directional (momentum collapse) ───────────────────────────────

    async def _open_directional(self, m: Market, side: str, price: float, now: float) -> None:
        """Open leg 1 of a directional trade: fade the dumping side."""
        cycle_id = f"dir-{m.condition_id[:12]}-{int(now)}"
        state = DirectionalState(
            cycle_id   = cycle_id,
            market_id  = m.condition_id,
            question   = m.question,
            asset      = self._infer_asset(m.question),
            leg1_side  = side,
            leg1_price = price,
            leg1_ts    = now,
        )
        self._active_dir[m.condition_id] = state

        # Opposite token is what we BUY (fading the dump)
        opposite_side   = "NO" if side == "YES" else "YES"
        opposite_token  = m.no_token if side == "YES" else m.yes_token

        log.info(
            "[DH15M-DIR] Dump detected on %s leg=%.3f — BUY %s | '%s'",
            side, price, opposite_side, m.question[:50],
        )

        await self._db.insert_dump_hedge_event(
            cycle_id    = cycle_id,
            market_id   = m.condition_id,
            question    = m.question,
            asset       = state.asset,
            phase       = "LEG1",
            leg1_side   = side,
            leg1_price  = price,
            leg2_price  = None,
            sum_price   = None,
            shares      = self._cfg.dh15m_shares,
            status      = "LEG1_FILLED",
            detail_json = json.dumps({"mode": "directional", "fade_side": opposite_side}),
        )

        if self._cfg.dry_run or opposite_token is None:
            return

        req = OrderRequest(
            token_id = opposite_token.token_id,
            price    = round(price, 2),
            size     = round(self._cfg.dh15m_shares, 2),
            side     = Side.BUY,
        )
        try:
            resp = await self._clob.place_order(req)
            if not resp.success:
                log.warning("[DH15M-DIR] Leg1 order failed: %s", resp.error_message)
        except Exception as exc:
            log.error("[DH15M-DIR] Leg1 exception: %s", exc)

    async def _manage_directional(
        self,
        state: DirectionalState,
        yes_ask: float,
        no_ask: float,
        now: float,
    ) -> Optional[str]:
        """Check if we can close the directional position for profit."""
        opposite_ask = no_ask if state.leg1_side == "YES" else yes_ask
        sum_price    = state.leg1_price + opposite_ask

        if sum_price <= self._cfg.dh15m_sum_target:
            # We can now sell the opposite leg at a profit
            pnl_est = (1.0 - sum_price) * self._cfg.dh15m_shares
            log.info(
                "[DH15M-DIR] HEDGED sum=%.3f est_pnl=+$%.4f | '%s'",
                sum_price, pnl_est, state.question[:50],
            )
            await self._db.insert_dump_hedge_event(
                cycle_id=state.cycle_id,
                market_id=state.market_id,
                question=state.question,
                asset=state.asset,
                phase="LEG2",
                leg1_side=state.leg1_side,
                leg1_price=state.leg1_price,
                leg2_price=opposite_ask,
                sum_price=sum_price,
                shares=self._cfg.dh15m_shares,
                status="HEDGED",
                detail_json=json.dumps({"mode": "directional", "est_pnl": round(pnl_est, 6)}),
            )
            del self._active_dir[state.market_id]
            self._session_dir_trades += 1
            self._session_pnl_est    += pnl_est
            return "HEDGED"

        wait_s = now - state.leg1_ts
        if wait_s >= self._cfg.dh15m_stop_hedge_wait_seconds:
            log.info(
                "[DH15M-DIR] STOP_HEDGE after %.0fs, sum=%.3f | '%s'",
                wait_s, sum_price, state.question[:50],
            )
            await self._db.insert_dump_hedge_event(
                cycle_id=state.cycle_id,
                market_id=state.market_id,
                question=state.question,
                asset=state.asset,
                phase="STOP_HEDGE",
                leg1_side=state.leg1_side,
                leg1_price=state.leg1_price,
                leg2_price=opposite_ask,
                sum_price=sum_price,
                shares=self._cfg.dh15m_shares,
                status="STOP_HEDGED",
                detail_json=json.dumps({"wait_seconds": round(wait_s, 2)}),
            )
            del self._active_dir[state.market_id]
            return "STOPPED"

        return None

    # ── Price history helpers ──────────────────────────────────────────────────

    def _append_price(self, key: str, ts: float, ask: float) -> None:
        hist = self._price_hist[key]
        hist.append((ts, ask))
        cutoff = ts - self._cfg.dh15m_window_seconds
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _detect_leg1(
        self, market_id: str, *, yes_ask: float, no_ask: float
    ) -> tuple[str, float] | None:
        """Detect a sharp dump on one leg and return (side, ask) of the DUMPED leg."""
        yes_drop = self._drop_ratio(f"{market_id}:YES", yes_ask)
        no_drop  = self._drop_ratio(f"{market_id}:NO",  no_ask)
        threshold = self._cfg.dh15m_move_threshold

        if yes_drop >= threshold and yes_drop >= no_drop:
            return "YES", yes_ask
        if no_drop >= threshold:
            return "NO", no_ask
        return None

    def _drop_ratio(self, key: str, current_ask: float) -> float:
        hist = self._price_hist[key]
        if not hist:
            return 0.0
        peak = max(p for _, p in hist)
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - current_ask) / peak)

    def _is_candidate_market(self, market: Market) -> bool:
        if not market.active or market.closed or market.archived:
            return False
        minutes_to_end = (market.days_to_end or 0.0) * 24.0 * 60.0
        if minutes_to_end < self._cfg.dh15m_min_minutes_to_end:
            return False
        if minutes_to_end > self._cfg.dh15m_max_minutes_to_end:
            return False
        q = (market.question or "").lower()
        if "up" not in q and "down" not in q:
            return False
        asset = self._infer_asset(market.question)
        if asset not in self._cfg.dh15m_assets_set:
            return False
        return True

    def _infer_asset(self, question: str) -> str:
        q = (question or "").lower()
        for asset in ("btc", "eth", "sol", "xrp"):
            if asset in q:
                return asset
        return "unknown"

    # ── Session stats ──────────────────────────────────────────────────────────

    def session_stats(self) -> dict:
        return {
            "arbs_executed":   self._session_arbs,
            "dir_trades":      self._session_dir_trades,
            "est_pnl_session": round(self._session_pnl_est, 4),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _noop_book():
    """Return None when a market is missing its YES/NO token."""
    return None
