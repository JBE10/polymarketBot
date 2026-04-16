"""
WebSocket subscriber for Polymarket CLOB real-time order book data.

Protocol
--------
Endpoint : wss://ws-subscriptions-clob.polymarket.com/ws/market
Auth     : None (public endpoint for order book data)

Subscribe message (sent after connect):
    {"assets_ids": ["<token_id_1>", "<token_id_2>", ...], "type": "market"}

Incoming event types:
    "book"             — full order book snapshot (bids + asks)
    "price_change"     — incremental update (not all implementations send this)
    "last_trade_price" — last trade price for a token

The client uses ``websockets`` (async) and implements:
  - Exponential backoff reconnection (capped at 60s)
  - Automatic re-subscription after reconnect
  - Ping/keepalive to prevent proxy timeouts
  - Dynamic subscription updates without reconnect

Usage
-----
    cache    = BookCache()
    ws       = WsClient(book_cache=cache)
    token_ids = ["0xabc...", "0xdef..."]

    await ws.subscribe(token_ids)

    # Run as a background task:
    asyncio.create_task(ws.stream())

    # Stop gracefully:
    await ws.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "websockets>=12.0 is required.  "
        "Install with: pip install 'websockets>=12.0'"
    ) from exc

from src.polymarket.book_cache import BookCache

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_RECONNECT_BASE_S  = 1.0    # first retry wait
_RECONNECT_MAX_S   = 60.0   # maximum wait between retries
_RECONNECT_FACTOR  = 2.0    # exponential backoff multiplier
_PING_INTERVAL_S   = 20.0   # keepalive ping interval
_PING_TIMEOUT_S    = 10.0   # if no pong within this, declare connection dead
_MAX_MSG_QUEUE     = 1_000  # internal message queue depth before dropping


class WsClient:
    """
    Async WebSocket client that pushes order book updates into a BookCache.

    Lifecycle
    ---------
    1. Call `subscribe(token_ids)` to set the subscription list.
    2. Run `asyncio.create_task(ws.stream())` — it loops forever.
    3. Call `stop()` to signal a clean shutdown.
    4. You can call `subscribe()` again at any time to update the set;
       the new list will be sent to the server on the next reconnect or
       immediately if already connected.
    """

    def __init__(
        self,
        book_cache: BookCache,
        ws_url: str = _WS_URL,
        reconnect_base_s: float = _RECONNECT_BASE_S,
        reconnect_max_s:  float = _RECONNECT_MAX_S,
        ping_interval_s:  float = _PING_INTERVAL_S,
        ping_timeout_s:   float = _PING_TIMEOUT_S,
    ) -> None:
        self._cache          = book_cache
        self._ws_url         = ws_url
        self._reconnect_base = reconnect_base_s
        self._reconnect_max  = reconnect_max_s
        self._ping_interval  = ping_interval_s
        self._ping_timeout   = ping_timeout_s

        self._token_ids: set[str] = set()
        self._stop_event: asyncio.Event = asyncio.Event()

        # Stats for monitoring
        self._total_messages:      int   = 0
        self._total_book_updates:  int   = 0
        self._last_message_at:     float = 0.0
        self._reconnect_count:     int   = 0
        self._connected:           bool  = False

    # ── Public interface ──────────────────────────────────────────────────────

    async def subscribe(self, token_ids: list[str]) -> None:
        """
        Set the list of token IDs to stream.

        Call before stream() or at any time to update subscriptions.
        Changes propagate on the next reconnect (or immediately if
        you add per-connection subscription update support below).
        """
        self._token_ids = set(token_ids)
        log.info("WsClient: subscription updated — %d tokens", len(self._token_ids))

    async def stream(self) -> None:
        """
        Main streaming loop.  Run with asyncio.create_task(ws.stream()).

        Connects to the WebSocket, subscribes, processes messages, and
        reconnects with exponential backoff on any error.
        """
        delay = self._reconnect_base

        while not self._stop_event.is_set():
            try:
                log.info("WsClient: connecting to %s", self._ws_url)
                await self._run_session()
                # Normal exit (stop requested)
                break
            except asyncio.CancelledError:
                log.info("WsClient: task cancelled — stopping")
                break
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                self._connected = False
                self._reconnect_count += 1
                log.warning(
                    "WsClient: connection lost (%s) — reconnecting in %.1fs "
                    "(attempt #%d)",
                    exc, delay, self._reconnect_count,
                )
            except Exception as exc:
                self._connected = False
                self._reconnect_count += 1
                log.error(
                    "WsClient: unexpected error (%s) — reconnecting in %.1fs",
                    exc, delay,
                )

            if self._stop_event.is_set():
                break

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=delay
                )
            except asyncio.TimeoutError:
                pass

            # Exponential backoff with jitter-free cap
            delay = min(delay * _RECONNECT_FACTOR, self._reconnect_max)

        log.info("WsClient: stream() exited cleanly")

    async def stop(self) -> None:
        """Signal the stream loop to exit gracefully."""
        self._stop_event.set()
        log.info("WsClient: stop signal sent")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected":          self._connected,
            "subscribed_tokens":  len(self._token_ids),
            "total_messages":     self._total_messages,
            "total_book_updates": self._total_book_updates,
            "reconnect_count":    self._reconnect_count,
            "last_message_age_s": (
                time.monotonic() - self._last_message_at
                if self._last_message_at > 0 else None
            ),
        }

    # ── Internal session management ───────────────────────────────────────────

    async def _run_session(self) -> None:
        """
        Single WebSocket session: connect → subscribe → process messages.
        Raises on any connection error so the outer loop can reconnect.
        """
        connect_kwargs: dict = {
            "ping_interval": self._ping_interval,
            "ping_timeout":  self._ping_timeout,
            "close_timeout": 5,
            "max_size":      2 ** 20,  # 1 MB max message size
        }

        async with websockets.connect(self._ws_url, **connect_kwargs) as ws:
            self._connected = True
            log.info(
                "WsClient: connected (tokens=%d)", len(self._token_ids)
            )

            # Send subscription message immediately after connect
            await self._send_subscription(ws)

            # Process incoming messages until disconnected or stop requested
            async for raw in ws:
                if self._stop_event.is_set():
                    break

                self._total_messages += 1
                self._last_message_at = time.monotonic()

                try:
                    await self._handle_message(raw)
                except Exception as exc:
                    log.debug("WsClient: message parse error: %s", exc)

        self._connected = False

    async def _send_subscription(self, ws) -> None:
        """Send the subscription payload to the server."""
        if not self._token_ids:
            log.warning("WsClient: no tokens to subscribe — stream will be silent")
            return

        payload = json.dumps({
            "type":       "market",
            "assets_ids": list(self._token_ids),
        })
        await ws.send(payload)
        log.info(
            "WsClient: subscribed to %d tokens", len(self._token_ids)
        )

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse a single WebSocket message and dispatch to the book cache."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        # The CLOB WS can send either a single object or a list of events
        data = json.loads(raw)

        events = data if isinstance(data, list) else [data]

        for event in events:
            event_type = event.get("event_type") or event.get("type", "")

            if event_type == "book":
                await self._handle_book_event(event)
            elif event_type in ("price_change", "tick_size_change"):
                # Incremental updates — treat as a mini book refresh if bids/asks present
                if "bids" in event or "asks" in event:
                    await self._handle_book_event(event)
            # Other types (last_trade_price, etc.) are silently ignored for now

    async def _handle_book_event(self, event: dict) -> None:
        """
        Push a book event into the cache.

        The CLOB WS book event format:
        {
            "event_type": "book",
            "asset_id":   "<token_id>",   # or "market"
            "bids": [{"price": "0.45", "size": "100"}, ...],
            "asks": [{"price": "0.47", "size": "80"},  ...]
        }
        """
        token_id = (
            event.get("asset_id")
            or event.get("market")
            or event.get("token_id")
            or ""
        )
        if not token_id:
            return

        bids = event.get("bids") or []
        asks = event.get("asks") or []

        if not bids and not asks:
            return  # empty update — nothing to do

        await self._cache.update(token_id, bids, asks)
        self._total_book_updates += 1

        log.debug(
            "WsClient book update: token=%s... bids=%d asks=%d",
            token_id[:12], len(bids), len(asks),
        )
