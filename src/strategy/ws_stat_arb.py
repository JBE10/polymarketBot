"""
ws_stat_arb.py — High-Frequency Cross-Market Arbitrage Engine

This engine listens to orderbook updates via WebSocket.
It groups markets into Clusters (same underlying and expiry, different strikes).
On every tick, it looks for:
1. Cross-Market Arbitrage (Strike Arbitrage): 
   P(BTC > 65k) MUST be >= P(BTC > 65.5k). If P(65.5k) is higher, we buy 65k YES and sell 65.5k YES (or buy 65.5k NO).
2. Internal Spread Arbitrage:
   YES_ask + NO_ask < 1.0 within the same market.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from src.core.config import Settings
from src.polymarket.clob_client import AsyncClobClient
from src.polymarket.ws_client import WsClient
from src.polymarket.book_cache import BookCache
from src.polymarket.models import OrderRequest, Side, OrderType, Market
from src.core.database import Database
import json

log = logging.getLogger(__name__)


class WsStatArbEngine:
    def __init__(
        self,
        cfg: Settings,
        clob: AsyncClobClient,
        ws: WsClient,
        cache: BookCache,
        db: Database,
    ) -> None:
        self._cfg = cfg
        self._clob = clob
        self._ws = ws
        self._cache = cache
        self._db = db
        
        self.running = False
        
        # internal stats
        self.cross_arb_count = 0
        self.spread_arb_count = 0
        self.last_log_time = 0.0

    async def initialize_clusters(self) -> None:
        """Fetch clusters and subscribe via WS."""
        log.info("WsStatArbEngine: Fetching BTC/ETH Market Clusters...")
        
        clusters = await self._clob.get_market_clusters(limit=500)
        
        if not clusters:
            log.warning("WsStatArbEngine: No valid market clusters found.")
            self._clusters = []
            return

        token_ids_to_sub = []
        self._clusters = clusters  # save for the loop

        for cluster in clusters:
            base_q = cluster["base_question"]
            markets = cluster["markets"]
            log.info(f"Subscribing to cluster: {base_q} ({len(markets)} strikes)")
            
            for strike, m in markets:
                for t in m.tokens:
                    if t.token_id:
                        token_ids_to_sub.append(t.token_id)
        
        # Suscribir WS
        await self._ws.subscribe(token_ids_to_sub)
        
    async def stop(self) -> None:
        self.running = False
        
    async def run(self) -> None:
        if not hasattr(self, '_clusters') or not self._clusters:
            log.warning("WsStatArbEngine: No clusters to monitor. Exiting loop.")
            return

        self.running = True
        log.info("WsStatArbEngine: Event loop listening to orderbook ticks...")
        # Ignorar actualizaciones del WS durante cooldown
        cooldown_until = 0.0
        
        while self.running:
            # Wake up specifically when WS pushes new books
            await self._cache.updated_event.wait()
            
            now = time.time()
            if now < cooldown_until:
                continue

            executed = await self._evaluate_clusters()
            
            if executed:
                # Add a 2s cooldown to prevent order spam
                cooldown_until = now + 2.0
                
            if now - self.last_log_time > 30.0:
                self.last_log_time = now
                log.info(
                    f"WsStatArbEngine Stats | Cross Arbs: {self.cross_arb_count} "
                    f"| Spread Arbs: {self.spread_arb_count} | WS Connected: {self._ws.is_connected}"
                )

    async def _evaluate_clusters(self) -> bool:
        """
        Calculates mathematical impossibilities across the cluster.
        Returns True if an order was placed.
        """
        for cluster in self._clusters:
            markets = cluster["markets"]
            
            # 1. Sort by strike (already done, but just to be sure)
            # Strike N < Strike N+1
            # Probability(YES for Strike N) >= Probability(YES for Strike N+1)
            
            # Iterate pairwise
            for i in range(len(markets) - 1):
                strike_low, m_low = markets[i]
                strike_high, m_high = markets[i+1]
                
                # Fetch books
                book_low_yes = self._get_token_book(m_low, "Yes")
                book_high_yes = self._get_token_book(m_high, "Yes")
                book_low_no = self._get_token_book(m_low, "No")
                book_high_no = self._get_token_book(m_high, "No")
                
                if not book_low_yes or not book_high_yes or not book_low_no or not book_high_no:
                    continue
                
                ask_low_yes = book_low_yes.best_ask
                bid_low_yes = book_low_yes.best_bid
                ask_high_yes = book_high_yes.best_ask
                bid_high_yes = book_high_yes.best_bid
                
                if ask_low_yes is None or bid_high_yes is None:
                    continue
                    
                # Cross-market mispricing (Strike Arbitrage):
                # If someone is buying the high strike at a price HIGHER than the low strike is being sold.
                # Example: Bid(BTC>65.5k) = 0.60, Ask(BTC>65.0k) = 0.50
                # We can Sell 65.5k at 0.60 and Buy 65.0k at 0.50! Guaranteed 0.10 profit.
                # To "Sell" 65.5k YES, we actually Buy 65.5k NO.
                
                spread_edge = bid_high_yes - ask_low_yes
                
                # We want edge > 0.02 to cover taker fees
                if spread_edge > self._cfg.dh15m_cost_buffer_per_pair:
                    log.warning(
                        f"🚨 STAT-ARB FOUND: {m_low.question[:30]}... "
                        f"Strike={strike_low} AskYes={ask_low_yes} | "
                        f"Strike={strike_high} BidYes={bid_high_yes} | Edge={spread_edge:.3f}"
                    )
                    
                    # Ejecutar Arb
                    # Buy Yes on Low Strike
                    # Sell Yes on High Strike (Buy No on High Strike)
                    ask_high_no = book_high_no.best_ask
                    if ask_high_no is not None and (ask_low_yes + ask_high_no < 1.0 - self._cfg.dh15m_cost_buffer_per_pair):
                         success = await self._execute_cross_arb(
                             m_low, "Yes", ask_low_yes,
                             m_high, "No", ask_high_no
                         )
                         if success:
                             self.cross_arb_count += 1
                             asyncio.create_task(self._log_event(
                                 market_id=m_low.condition_id,
                                 question=m_low.question,
                                 phase="CROSS_ARB",
                                 detail={"type": "strike_inversion", "edge": spread_edge, "s1_yes": ask_low_yes, "s2_no": ask_high_no}
                             ))
                             return True
                             
                # 2. Internal Spread Arb 
                ask_low_no = book_low_no.best_ask
                if ask_low_no is not None and ask_low_yes + ask_low_no <= self._cfg.dh15m_sum_target:
                    log.warning(f"🚨 SPREAD-ARB FOUND: {m_low.question[:30]} Yes={ask_low_yes} No={ask_low_no}")
                    success = await self._execute_cross_arb(
                        m_low, "Yes", ask_low_yes,
                        m_low, "No", ask_low_no
                    )
                    if success:
                        self.spread_arb_count += 1
                        asyncio.create_task(self._log_event(
                            market_id=m_low.condition_id,
                            question=m_low.question,
                            phase="SPREAD_ARB",
                            detail={"type": "spread", "edge": 1.0 - (ask_low_yes + ask_low_no), "yes": ask_low_yes, "no": ask_low_no}
                        ))
                        return True
                        
        return False

    def _get_token_book(self, market: Market, outcome: str):
        for t in market.tokens:
            if t.outcome == outcome:
                return self._cache.get(t.token_id)
        return None

    async def _execute_cross_arb(self, m1, outcome1, price1, m2, outcome2, price2) -> bool:
        if self._cfg.dry_run:
            log.info(f"[DRY-RUN] Executing arb: Buy {outcome1} @ {price1} and Buy {outcome2} @ {price2}")
            return True
            
        t1 = next((t for t in m1.tokens if t.outcome == outcome1), None)
        t2 = next((t for t in m2.tokens if t.outcome == outcome2), None)
        
        if not t1 or not t2: return False
        
        shares = self._cfg.dh15m_shares
        
        try:
            req1 = OrderRequest(token_id=t1.token_id, price=price1, size=shares, side=Side.BUY, order_type=OrderType.FOK)
            req2 = OrderRequest(token_id=t2.token_id, price=price2, size=shares, side=Side.BUY, order_type=OrderType.FOK)
            
            res1, res2 = await asyncio.gather(
                self._clob.place_order(req1),
                self._clob.place_order(req2)
            )
            
            log.info(f"Cross Arb Executed: Leg1={res1.status.value}, Leg2={res2.status.value}")
            return res1.status.value == "matched" and res2.status.value == "matched"
        except Exception as e:
            log.error(f"Error executing cross arb: {e}")
            return False

    async def _log_event(self, market_id: str, question: str, phase: str, detail: dict) -> None:
        try:
            await self._db.insert_dump_hedge_event(
                cycle_id=str(int(time.time())),
                market_id=market_id,
                question=question,
                asset="ARB",
                phase=phase,
                leg1_side="YES",
                leg1_price=None,
                leg2_price=None,
                sum_price=None,
                shares=self._cfg.dh15m_shares,
                status="EXECUTED",
                detail_json=json.dumps(detail)
            )
        except Exception as e:
            log.error(f"Failed to log dump_hedge_event: {e}")
