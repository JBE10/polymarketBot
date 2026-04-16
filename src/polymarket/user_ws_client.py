"""
UserWsClient — Authenticated WebSocket for Polymarket CLOB user events.

This client connects to the **private** user channel, which sends push
notifications for every state change on YOUR orders:

  wss://ws-subscriptions-clob.polymarket.com/ws/user

Authentication
--------------
The user channel requires an HMAC-signed subscription message:

    {
        "auth": {
            "apiKey":      "<L2 API key>",
            "secret":      "<L2 API secret>",
            "passphrase":  "<L2 API passphrase>"
        },
        "markets": ["<condition_id_1>", ...],
        "type": "user"
    }

The HMAC signature uses the L2 API secret (NOT the private key) and is
already handled by the py-clob-client ApiCreds object — we just pass the
fields directly.

Incoming event types (relevant ones)
-------------------------------------
  "order"   — order lifecycle: OPEN, PARTIAL_FILLED, MATCHED, CANCELLED
  "trade"   — a trade was matched (both maker and taker get this)
  "error"   — protocol/auth error

Each "order" event looks like:
    {
        "event_type":    "order",
        "id":            "<order_id>",
        "type":          "MATCHED",          # OPEN | PARTIAL_FILLED | MATCHED | CANCELLED
        "size_matched":  "50.00",
        "price":         "0.46",
        "side":          "BUY",
        "asset_id":      "<token_id>",
        "market":        "<condition_id>",
        "timestamp":     "1713275948123"
    }

The client pushes parsed events into a FillCache.  The MarketMaker reads
from FillCache instead of polling get_open_orders() via HTTP.

Usage
-----
    fill_cache   = FillCache()
    user_ws      = UserWsClient(fill_cache=fill_cache, api_creds=creds)
    await user_ws.subscribe(market_ids=["0xabc..."])
    asyncio.create_task(user_ws.stream())
    await user_ws.stop()   # graceful shutdown
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "websockets>=12.0 is required — pip install 'websockets>=12.0'"
    ) from exc

from src.polymarket.fill_cache import FillCache, FillStatus

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

_RECONNECT_BASE_S  = 1.0
_RECONNECT_MAX_S   = 60.0
_RECONNECT_FACTOR  = 2.0
_PING_INTERVAL_S   = 20.0
_PING_TIMEOUT_S    = 10.0

# CLOB order status strings → FillStatus mapping
_STATUS_MAP: dict[str, FillStatus] = {
    "OPEN":           FillStatus.OPEN,
    "LIVE":           FillStatus.OPEN,
    "PARTIAL_FILLED": FillStatus.PARTIAL,
    "MATCHED":        FillStatus.FILLED,
    "FILLED":         FillStatus.FILLED,
    "CANCELLED":      FillStatus.CANCELLED,
    "CANCELED":       FillStatus.CANCELLED,  # alternate spelling
    "EXPIRED":        FillStatus.CANCELLED,
    "CANCELLED_PARTIAL": FillStatus.CANCELLED,
}


@dataclass(frozen=True)
class ApiCreds:
    """L2 API credentials from py-clob-client (or derived manually)."""
    api_key:        str
    api_secret:     str
    api_passphrase: str


class UserWsClient:
    """
    Authenticated WebSocket client for Polymarket user order events.

    Pushes fill/cancel events into a FillCache so MarketMaker can detect
    fills without calling get_open_orders() via HTTP.

    Lifecycle
    ---------
    1. Call `subscribe(market_ids)` to set the market condition IDs to watch.
    2. Run `asyncio.create_task(user_ws.stream())` — it loops forever.
    3. MarketMaker reads `fill_cache.is_filled(order_id)` for fill detection.
    4. Call `stop()` to exit gracefully.
    """

    def __init__(
        self,
        fill_cache:       FillCache,
        api_creds:        ApiCreds,
        ws_url:           str   = _USER_WS_URL,
        reconnect_base_s: float = _RECONNECT_BASE_S,
        reconnect_max_s:  float = _RECONNECT_MAX_S,
        ping_interval_s:  float = _PING_INTERVAL_S,
        ping_timeout_s:   float = _PING_TIMEOUT_S,
    ) -> None:
        self._cache          = fill_cache
        self._creds          = api_creds
        self._ws_url         = ws_url
        self._reconnect_base = reconnect_base_s
        self._reconnect_max  = reconnect_max_s
        self._ping_interval  = ping_interval_s
        self._ping_timeout   = ping_timeout_s

        self._market_ids: set[str] = set()
        self._stop_event: asyncio.Event = asyncio.Event()

        # Stats
        self._total_messages: int   = 0
        self._total_fills:    int   = 0
        self._total_cancels:  int   = 0
        self._reconnect_count: int  = 0
        self._connected:      bool  = False
        self._last_message_at: float = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Set the markets to watch for order events.
        Must be called before stream() or when the approved market set changes.
        Changes propagate on the next reconnect.
        """
        self._market_ids = set(market_ids)
        log.info("UserWsClient: subscription updated — %d markets", len(self._market_ids))

    async def stream(self) -> None:
        """
        Main user channel streaming loop.
        Run with asyncio.create_task(user_ws.stream()).
        """
        delay = self._reconnect_base

        while not self._stop_event.is_set():
            try:
                log.info(
                    "UserWsClient: connecting to %s (%d markets)",
                    self._ws_url, len(self._market_ids),
                )
                await self._run_session()
                break  # normal exit on stop()
            except asyncio.CancelledError:
                break
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                self._connected = False
                self._reconnect_count += 1
                log.warning(
                    "UserWsClient: connection lost (%s) — reconnect in %.1fs (#%d)",
                    exc, delay, self._reconnect_count,
                )
            except Exception as exc:
                self._connected = False
                self._reconnect_count += 1
                log.error(
                    "UserWsClient: unexpected error (%s) — reconnect in %.1fs",
                    exc, delay,
                )

            if self._stop_event.is_set():
                break
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * _RECONNECT_FACTOR, self._reconnect_max)

        log.info("UserWsClient: stream() exited cleanly")

    async def stop(self) -> None:
        self._stop_event.set()
        log.info("UserWsClient: stop signal sent")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected":       self._connected,
            "watched_markets": len(self._market_ids),
            "total_messages":  self._total_messages,
            "total_fills":     self._total_fills,
            "total_cancels":   self._total_cancels,
            "reconnect_count": self._reconnect_count,
            "last_msg_age_s":  (
                time.monotonic() - self._last_message_at
                if self._last_message_at > 0 else None
            ),
        }

    # ── Session management ────────────────────────────────────────────────────

    async def _run_session(self) -> None:
        connect_kwargs = {
            "ping_interval": self._ping_interval,
            "ping_timeout":  self._ping_timeout,
            "close_timeout": 5,
            "max_size":      2 ** 20,
        }

        async with websockets.connect(self._ws_url, **connect_kwargs) as ws:
            self._connected = True
            log.info(
                "UserWsClient: connected (%d markets)", len(self._market_ids)
            )

            await self._send_auth_subscription(ws)

            async for raw in ws:
                if self._stop_event.is_set():
                    break

                self._total_messages += 1
                self._last_message_at = time.monotonic()

                try:
                    await self._handle_message(raw)
                except Exception as exc:
                    log.debug("UserWsClient: parse error: %s", exc)

        self._connected = False

    async def _send_auth_subscription(self, ws) -> None:
        """Send the authenticated subscription message."""
        payload = json.dumps({
            "type":    "user",
            "auth": {
                "apiKey":     self._creds.api_key,
                "secret":     self._creds.api_secret,
                "passphrase": self._creds.api_passphrase,
            },
            "markets": list(self._market_ids),
        })
        await ws.send(payload)
        log.info(
            "UserWsClient: auth subscription sent (%d markets)",
            len(self._market_ids),
        )

    # ── Message handling ──────────────────────────────────────────────────────

    async def _handle_message(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        data = json.loads(raw)
        events = data if isinstance(data, list) else [data]

        for event in events:
            event_type = event.get("event_type") or event.get("type", "")

            if event_type == "order":
                await self._handle_order_event(event)
            elif event_type == "trade":
                await self._handle_trade_event(event)
            elif event_type == "error":
                log.error(
                    "UserWsClient server error: %s",
                    event.get("message", event),
                )

    async def _handle_order_event(self, event: dict) -> None:
        """
        Parse an 'order' event and push the result into FillCache.

        Event format:
            {
                "event_type": "order",
                "id":         "<order_id>",
                "type":       "MATCHED",
                "size_matched": "50.00",
                "price":      "0.46",
                "side":       "BUY",
                "asset_id":   "<token_id>",
                "market":     "<condition_id>"
            }
        """
        order_id = event.get("id") or event.get("order_id") or ""
        if not order_id:
            log.debug("UserWsClient: order event with no id — skipping")
            return

        raw_status  = (event.get("type") or event.get("status") or "").upper()
        status      = _STATUS_MAP.get(raw_status, FillStatus.UNKNOWN)

        size_matched = _safe_float(event.get("size_matched") or event.get("original_size"))
        price        = _safe_float(event.get("price") or event.get("limit_price"))
        side         = (event.get("side") or "").upper()
        token_id     = event.get("asset_id") or event.get("token_id") or ""

        await self._cache.record_event(
            order_id=order_id,
            status=status,
            size_matched=size_matched,
            price=price,
            side=side,
            token_id=token_id,
        )

        if status == FillStatus.FILLED:
            self._total_fills += 1
            log.info(
                "UserWsClient: FILL order_id=%s side=%s price=%.3f size=%.0f",
                order_id[:16], side, price, size_matched,
            )
        elif status == FillStatus.CANCELLED:
            self._total_cancels += 1
            log.info(
                "UserWsClient: CANCEL order_id=%s", order_id[:16]
            )
        else:
            log.debug(
                "UserWsClient: order update order_id=%s status=%s",
                order_id[:16], status.value,
            )

    async def _handle_trade_event(self, event: dict) -> None:
        """
        Parse a 'trade' event.

        A trade event may arrive BEFORE the corresponding 'order' event is
        updated to MATCHED status.  We use it as a confirmatory signal:
        if we see a trade for one of our order IDs, mark it filled.

        Trade event format:
            {
                "event_type": "trade",
                "maker_order_id": "<id>",
                "taker_order_id": "<id>",
                "size":           "50.00",
                "price":          "0.46",
                "asset_id":       "<token_id>",
                "side":           "BUY"
            }
        """
        price    = _safe_float(event.get("price"))
        size     = _safe_float(event.get("size"))
        token_id = event.get("asset_id") or ""
        side     = (event.get("side") or "").upper()

        for key in ("maker_order_id", "taker_order_id"):
            oid = event.get(key) or ""
            if not oid:
                continue

            existing = self._cache.get_event(oid)
            if existing is None:
                # First time we hear about this order — record as filled from trade
                await self._cache.record_event(
                    order_id=oid,
                    status=FillStatus.FILLED,
                    size_matched=size,
                    price=price,
                    side=side,
                    token_id=token_id,
                )
                self._total_fills += 1
                log.info(
                    "UserWsClient: TRADE fill (first event) order_id=%s price=%.3f size=%.0f",
                    oid[:16], price, size,
                )
            elif existing.status != FillStatus.FILLED:
                # Upgrade known order to FILLED
                await self._cache.record_event(
                    order_id=oid,
                    status=FillStatus.FILLED,
                    size_matched=size,
                    price=price,
                    side=side,
                    token_id=token_id,
                )
                self._total_fills += 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def make_api_creds_from_clob(client) -> Optional[ApiCreds]:
    """
    Extract L2 API credentials from an initialized AsyncClobClient.

    Tries to pull them from the underlying ClobClient object.
    Returns None if credentials are not available (e.g. dry-run mode).
    """
    try:
        inner = client._client  # type: ignore[attr-defined]
        if inner is None:
            return None
        creds = inner.creds
        if creds is None:
            return None
        return ApiCreds(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
        )
    except AttributeError:
        return None
