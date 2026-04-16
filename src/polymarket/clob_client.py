"""
Async wrapper around the synchronous py-clob-client SDK.

The Polymarket CLOB SDK is blocking; every call is run in a thread-pool
executor via asyncio.to_thread() so it never blocks the event loop.

Authentication flow
-------------------
1. A private key is injected at construction time (from Secure Enclave).
2. `initialize()` derives L1 ECDSA credentials and L2 API credentials in one
   call using `create_or_derive_api_creds()`.
3. All order-placement calls require the L2 API creds to be set.

USDC approval (one-time setup)
------------------------------
Before the bot can place orders, the trading wallet must approve the CTF
Exchange contract to spend USDC on its behalf.  This is a one-time on-chain
transaction that the user runs manually:

    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
    usdc = w3.eth.contract(
        address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        abi=[{"constant":False,"inputs":[{"name":"spender","type":"address"},
             {"name":"amount","type":"uint256"}],"name":"approve",
             "outputs":[{"name":"","type":"bool"}],"type":"function"}],
    )
    tx = usdc.functions.approve(
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # CTF Exchange
        2**256 - 1,                                       # max approval
    ).build_transaction({...})

Reference: https://github.com/Polymarket/py-clob-client
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, Optional

import httpx

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OpenOrderParams,
    OrderArgs,
    OrderType,
    TradeParams,
)
from py_clob_client.constants import POLYGON

from src.polymarket.models import (
    Market,
    MarketToken,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    PriceLevel,
    Side,
)

log = logging.getLogger(__name__)


class AsyncClobClient:
    """
    Async façade over ClobClient.

    All public methods are coroutines that delegate to the synchronous SDK
    via a thread-pool so the event loop is never blocked.
    """

    def __init__(
        self,
        host: str,
        private_key: str,
        chain_id: int = POLYGON,
        dry_run: bool = True,
    ) -> None:
        self._host        = host
        self._private_key = private_key
        self._chain_id    = chain_id
        self._dry_run     = dry_run
        self._client: Optional[ClobClient] = None
        self._creds_ready = False
        # Async HTTP client para el CLOB (order book, trades, orders)
        self._http = httpx.AsyncClient(
            base_url=host,
            headers={"User-Agent": "polymarket-bot/1.0", "Accept": "application/json"},
            timeout=30.0,
        )
        # Async HTTP client para la Gamma API (market discovery con liquidez y volumen)
        self._gamma = httpx.AsyncClient(
            base_url="https://gamma-api.polymarket.com",
            headers={"User-Agent": "polymarket-bot/1.0", "Accept": "application/json"},
            timeout=30.0,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build the ClobClient.  En dry-run se omite la derivación de credenciales L2
        (que requiere red y una clave real) porque solo se leen mercados, no se
        colocan órdenes.
        """
        await asyncio.to_thread(self._sync_init)
        log.info(
            "CLOB client ready (host=%s, chain=%d, creds=%s)",
            self._host, self._chain_id,
            "OK" if self._creds_ready else "skipped (dry-run)",
        )

    def _sync_init(self) -> None:
        self._client = ClobClient(
            host=self._host,
            key=self._private_key,
            chain_id=self._chain_id,
            signature_type=0,       # EOA account (ECDSA); use 1 for Gnosis Safe
        )

        if self._dry_run:
            # En dry-run no derivamos credenciales L2 — solo leemos datos públicos
            log.info("CLOB: modo dry-run, credenciales L2 omitidas.")
            return

        try:
            creds: ApiCreds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)
            self._creds_ready = True
        except Exception as exc:
            log.warning(
                "No se pudieron derivar credenciales L2 del CLOB: %s\n"
                "Las consultas de mercado funcionarán, pero no se podrán colocar órdenes.",
                exc,
            )

    async def close(self) -> None:
        await self._http.aclose()
        await self._gamma.aclose()
        self._client = None

    # ── Market data ────────────────────────────────────────────────────────────
    # get_markets usa Gamma API (liquidez + volumen); order book usa CLOB API.

    async def get_markets(
        self,
        limit: int = 100,
    ) -> list[Market]:
        """
        Fetch active markets ordenados por volumen 24h (Gamma API).
        La Gamma API incluye liquidez, volume24hr y outcomePrices, que el CLOB
        no proporciona. Se descarta cualquier mercado cerrado o archivado.
        """
        try:
            resp = await self._gamma.get(
                "/markets",
                params={
                    "limit": limit,
                    "active":     "true",
                    "closed":     "false",
                    "archived":   "false",
                    "order":      "volume24hr",
                    "ascending":  "false",
                },
            )
            resp.raise_for_status()
            raw_list = resp.json()  # Gamma devuelve lista directa, no dict con "data"
        except Exception as exc:
            log.warning("get_markets (Gamma) HTTP error: %s", exc)
            return []
        markets = []
        for item in (raw_list if isinstance(raw_list, list) else []):
            m = _parse_gamma_market(item)
            if m:
                markets.append(m)
        return markets

    async def get_market_clusters(
        self,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Fetch active markets and group them into clusters of related strikes.
        Returns a list of dicts: {"base_question": str, "end_date_iso": str, "markets": list[(strike, Market)]}
        Only returns clusters with >= 2 strikes.
        """
        import re
        from collections import defaultdict
        
        markets = await self.get_markets(limit=limit)
        clusters = defaultdict(list)
        
        for m in markets:
            if not m.active or m.closed or m.archived: continue
            q = m.question or ""
            # Buscar valor de strike en dolares (ej. $65,000 o $1.25)
            m_dollars = re.search(r'\$([0-9,\.]+)', q)
            if m_dollars:
                strike_str = m_dollars.group(1).replace(',', '')
                try:
                    strike = float(strike_str)
                except ValueError:
                    continue
                # base_q es la pregunta sin el strike, ej "Will BTC be above $XXX at..."
                base_q = q.replace(f'${m_dollars.group(1)}', '$XXX')
                
                # Agrupar por la base_q y dias_restantes (para asegurar misma fecha)
                end_date_key = round(m.days_to_end or 0, 3)
                clusters[(base_q, end_date_key)].append((strike, m))
                
        valid_clusters = []
        for (base_q, _), strikes in clusters.items():
            if len(strikes) >= 2:
                # Ordenar de menor strike a mayor strike
                sorted_strikes = sorted(strikes, key=lambda x: x[0])
                valid_clusters.append({
                    "base_question": base_q,
                    "markets": sorted_strikes
                })
                
        return valid_clusters

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market by condition ID (Gamma API)."""
        try:
            resp = await self._gamma.get(
                "/markets",
                params={"condition_id": condition_id, "limit": 1},
            )
            resp.raise_for_status()
            items = resp.json()
            if isinstance(items, list) and items:
                return _parse_gamma_market(items[0])
        except Exception as exc:
            log.warning("get_market(%s) failed: %s", condition_id, exc)
        return None

    async def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch the live order book via direct async HTTP."""
        try:
            resp = await self._http.get("/book", params={"token_id": token_id})
            resp.raise_for_status()
            return _parse_order_book(resp.json())
        except Exception as exc:
            log.warning("get_order_book(%s) failed: %s", token_id, exc)
            return None

    async def get_price(self, token_id: str, side: Side) -> Optional[float]:
        """Fetch the current best price via direct async HTTP."""
        try:
            resp = await self._http.get(
                "/price", params={"token_id": token_id, "side": side.value}
            )
            resp.raise_for_status()
            return float(resp.json().get("price", 0) or 0)
        except Exception as exc:
            log.warning("get_price(%s) failed: %s", token_id, exc)
            return None

    async def get_trades(self, token_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch recent trades via CLOB, with Gamma API fallback.

        The CLOB ``/trades`` endpoint requires authentication and returns 401
        in dry-run mode (no L2 credentials).  When that happens we fall back
        to the Gamma API's public ``/prices`` endpoint which provides recent
        price history for the regime filter without needing authentication.
        """
        # --- Primary: CLOB /trades -------------------------------------------
        try:
            resp = await self._http.get(
                "/trades", params={"asset_id": token_id, "limit": limit}
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            if data:
                return data
        except Exception as exc:
            log.debug("CLOB /trades failed (expected in dry-run): %s", exc)

        # --- Fallback: Gamma API /prices -------------------------------------
        return await self._get_trades_gamma_fallback(token_id, limit)

    async def _get_trades_gamma_fallback(
        self, token_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Synthesize a trade-like price list from the Gamma /prices endpoint.

        The Gamma API endpoint ``/prices?token_id=X&fidelity=1&interval=all``
        returns historical price snapshots that are sufficient for the regime
        filter's Bollinger / ADX calculations.
        """
        try:
            resp = await self._gamma.get(
                "/prices",
                params={"token_id": token_id, "fidelity": 1, "interval": "all"},
            )
            resp.raise_for_status()
            raw = resp.json()

            # Gamma /prices returns {"history": [{"t": epoch, "p": price}, ...]}
            history = raw.get("history") or []
            if not history:
                log.debug("Gamma /prices returned empty history for %s", token_id)
                return []

            # Convert to trade-like dicts that the regime filter can consume
            trades = [
                {"price": str(point.get("p", 0)), "timestamp": point.get("t", 0)}
                for point in history[-limit:]
            ]
            log.debug(
                "Gamma fallback: %d price points for %s", len(trades), token_id[:12]
            )
            return trades

        except Exception as exc:
            log.warning("Gamma /prices fallback failed for %s: %s", token_id, exc)
            return []

    # ── Order placement ───────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Sign and submit a limit order to the CLOB.
        Returns OrderResponse — caller should inspect `.success`.
        """
        assert self._client
        try:
            order_args = OrderArgs(
                token_id=request.token_id,
                price=request.price,
                size=request.size,
                side=request.side.value,
            )
            order_type = (
                OrderType.FOK if request.order_type.value == "FOK"
                else OrderType.GTC
            )

            # Sign (CPU-bound but fast)
            signed = await asyncio.to_thread(
                partial(self._client.create_order, order_args=order_args)
            )
            # Submit
            resp = await asyncio.to_thread(
                partial(self._client.post_order, order=signed, order_type=order_type)
            )

            oid = resp.get("orderID") or resp.get("id")
            txh = resp.get("transactionHash")
            status_str = resp.get("status", "PENDING").upper()

            return OrderResponse(
                order_id=oid,
                status=OrderStatus(status_str) if status_str in OrderStatus.__members__ else OrderStatus.PENDING,
                transaction_hash=txh,
                filled_size=float(resp.get("sizeMatched", 0) or 0),
            )

        except Exception as exc:
            log.error("place_order failed: %s", exc)
            return OrderResponse(
                status=OrderStatus.REJECTED,
                error_message=str(exc),
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        assert self._client
        try:
            await asyncio.to_thread(
                partial(self._client.cancel, order_id=order_id)
            )
            return True
        except Exception as exc:
            log.warning("cancel_order(%s) failed: %s", order_id, exc)
            return False

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch all open orders for the authenticated address."""
        assert self._client
        try:
            raw = await asyncio.to_thread(
                partial(
                    self._client.get_open_orders,
                    params=OpenOrderParams(),
                )
            )
            return raw.get("data") or []
        except Exception as exc:
            log.warning("get_open_orders failed: %s", exc)
            return []

    async def get_balance_allowance(self) -> dict[str, float]:
        """Return USDC balance and CLOB allowance for the trading wallet."""
        assert self._client
        try:
            raw = await asyncio.to_thread(self._client.get_balance_allowance)
            return {
                "balance":   float(raw.get("balance",   0) or 0),
                "allowance": float(raw.get("allowance", 0) or 0),
            }
        except Exception as exc:
            log.warning("get_balance_allowance failed: %s", exc)
            return {"balance": 0.0, "allowance": 0.0}


# ── Parsing helpers ────────────────────────────────────────────────────────────

import json as _json


def _parse_gamma_market(raw: dict[str, Any]) -> Optional[Market]:
    """
    Convert a Gamma API market dict to a typed Market model.

    Gamma uses camelCase field names and stores outcome prices as a JSON-encoded
    array of strings, e.g. outcomePrices = '["0.82", "0.18"]'.
    CLOB token IDs live in clobTokenIds (also sometimes a JSON string).
    """
    try:
        # ── Token IDs ──────────────────────────────────────────────────────────
        raw_ids = raw.get("clobTokenIds") or []
        if isinstance(raw_ids, str):
            try:
                raw_ids = _json.loads(raw_ids)
            except Exception:
                raw_ids = []

        outcomes_list: list[str] = raw.get("outcomes") or []

        raw_prices = raw.get("outcomePrices") or []
        if isinstance(raw_prices, str):
            try:
                raw_prices = _json.loads(raw_prices)
            except Exception:
                raw_prices = []

        tokens = []
        for i, tid in enumerate(raw_ids):
            outcome = outcomes_list[i] if i < len(outcomes_list) else f"Outcome {i}"
            price   = float(raw_prices[i]) if i < len(raw_prices) else 0.0
            tokens.append(
                MarketToken(
                    token_id=str(tid),
                    outcome=outcome,
                    price=price,
                    winner=False,
                )
            )

        # ── Tags ───────────────────────────────────────────────────────────────
        tags_raw = raw.get("tags") or []
        tags: list[str] = []
        for t in tags_raw:
            if isinstance(t, str):
                tags.append(t)
            elif isinstance(t, dict):
                tags.append(t.get("label") or t.get("id") or "")

        return Market(
            condition_id=raw.get("conditionId") or raw.get("condition_id", ""),
            question_id=raw.get("questionID") or raw.get("question_id", ""),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            market_slug=raw.get("slug") or raw.get("market_slug", ""),
            end_date_iso=raw.get("endDateIso") or raw.get("endDate") or raw.get("end_date_iso"),
            start_date_iso=raw.get("startDateIso") or raw.get("startDate") or raw.get("start_date_iso"),
            active=bool(raw.get("active", True)),
            closed=bool(raw.get("closed", False)),
            archived=bool(raw.get("archived", False)),
            minimum_order_size=float(raw.get("orderMinSize") or raw.get("minimum_order_size", 1.0) or 1.0),
            minimum_tick_size=float(raw.get("orderPriceMinTickSize") or raw.get("minimum_tick_size", 0.01) or 0.01),
            tokens=tokens,
            tags=tags,
            volume=float(raw.get("volume", 0) or raw.get("volumeNum", 0) or 0),
            volume_24hr=float(raw.get("volume24hr", 0) or raw.get("volume24hrClob", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or raw.get("liquidityNum", 0) or 0),
        )
    except Exception as exc:
        log.debug("Failed to parse gamma market: %s — %s", raw.get("conditionId"), exc)
        return None


def _parse_market(raw: dict[str, Any]) -> Optional[Market]:
    """
    Fallback parser for CLOB API market dicts (used by get_market via CLOB endpoint).
    For most market discovery, prefer _parse_gamma_market.
    """
    try:
        tokens = []
        for t in raw.get("tokens") or []:
            tokens.append(
                MarketToken(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0) or 0),
                    winner=bool(t.get("winner", False)),
                )
            )
        return Market(
            condition_id=raw.get("condition_id", ""),
            question_id=raw.get("question_id", ""),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            market_slug=raw.get("market_slug", ""),
            end_date_iso=raw.get("end_date_iso") or raw.get("endDateIso"),
            start_date_iso=raw.get("start_date_iso") or raw.get("startDateIso"),
            active=bool(raw.get("active", True)),
            closed=bool(raw.get("closed", False)),
            archived=bool(raw.get("archived", False)),
            minimum_order_size=float(raw.get("minimum_order_size", 1.0) or 1.0),
            minimum_tick_size=float(raw.get("minimum_tick_size", 0.01) or 0.01),
            tokens=tokens,
            tags=raw.get("tags") or [],
            volume=float(raw.get("volume", 0) or 0),
            volume_24hr=float(raw.get("volume_24hr", 0) or raw.get("volume24hr", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
        )
    except Exception as exc:
        log.debug("Failed to parse market: %s — %s", raw.get("condition_id"), exc)
        return None


def _parse_order_book(raw: dict[str, Any]) -> OrderBook:
    def _levels(lst: list[dict]) -> list[PriceLevel]:
        out = []
        for item in lst or []:
            try:
                out.append(PriceLevel(price=float(item["price"]), size=float(item["size"])))
            except (KeyError, ValueError, TypeError):
                continue
        return out

    return OrderBook(
        market=raw.get("market", ""),
        asset_id=raw.get("asset_id", ""),
        bids=sorted(_levels(raw.get("bids", [])), key=lambda x: x.price, reverse=True),
        asks=sorted(_levels(raw.get("asks", [])), key=lambda x: x.price),
    )
