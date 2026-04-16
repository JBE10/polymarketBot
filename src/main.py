"""
Async entry point for the Polymarket Intelligence Bot.

Startup sequence
----------------
1.  Load settings from .env / environment.
2.  Retrieve the private key from macOS Keychain (Secure Enclave-backed).
3.  Connect to the SQLite database (WAL mode).
4.  Probe Polygon RPC endpoints and establish AsyncWeb3.
5.  Initialise the Polymarket CLOB client and derive API credentials.
6.  Start the ChromaDB / LlamaIndex RAG engine.
7.  Start the WebSocket order book feed (event-driven, replaces HTTP polling).
8.  Start the Brain analysis engine and Trigger execution engine.
9.  Enter the (now lightweight) LLM discovery loop for market selection.
10. On SIGINT/SIGTERM: drain in-flight tasks, close connections, exit cleanly.

Run
---
    python -m src.main                      # from repo root
    DRY_RUN=true python -m src.main         # override .env inline
"""
from __future__ import annotations

import asyncio
import logging
import signal
import socket
import sys
from pathlib import Path
from typing import Optional

# ── uvloop: faster event loop (2-4× I/O throughput on Linux/macOS) ──────────
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # fallback to standard asyncio — install uvloop for production

# Ensure the repo root is on the path so `src.*` imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging.handlers

from src.ai.rag_engine import RagEngine
from src.core.brain import Brain
from src.core.config import get_settings
from src.core.database import Database
from src.core.notifier import notifier
from src.core.provider import PolyProvider
from src.core.security import get_private_key
from src.polymarket.book_cache import BookCache
from src.polymarket.clob_client import AsyncClobClient
from src.polymarket.fill_cache import FillCache
from src.polymarket.models import Market
from src.polymarket.user_ws_client import UserWsClient, make_api_creds_from_clob
from src.polymarket.ws_client import WsClient
from src.strategy.ws_stat_arb import WsStatArbEngine
from src.strategy.llm_evaluator import LLMEvaluator
from src.strategy.market_maker import MarketMaker

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(cfg) -> None:
    # Ensure logs directory exists
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                filename=cfg.log_file,
                maxBytes=cfg.log_max_bytes,
                backupCount=cfg.log_backup_count,
            ),
        ],
    )
    # Silence chatty third-party loggers — only show warnings and above
    for _noisy in ("httpx", "httpcore", "websockets", "websockets.client",
                   "asyncio", "chromadb", "chromadb.telemetry"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("polymarket_bot")


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown_event: asyncio.Event = asyncio.Event()


def _install_signal_handlers() -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal, sig)


def _on_signal(sig: signal.Signals) -> None:
    log.warning("Received %s — initiating graceful shutdown…", sig.name)
    _shutdown_event.set()


# ── Dual loops ────────────────────────────────────────────────────────────────

_DRY_RUN_BANKROLL = 1_000.0


async def _get_bankroll(provider, cfg) -> float:
    """Fetch USDC balance or use simulated bankroll in dry-run."""
    bankroll = 0.0
    if cfg.polymarket_wallet_address:
        try:
            bankroll = await provider.get_usdc_balance(cfg.polymarket_wallet_address)
            log.info("USDC balance: $%.2f", bankroll)
        except Exception as exc:
            log.warning("Could not fetch USDC balance: %s — using bankroll=0", exc)

    if bankroll <= 0 and not cfg.dry_run:
        log.warning("Bankroll is $0.  Set DRY_RUN=true or fund the wallet.")

    if cfg.dry_run and bankroll < 1.0:
        bankroll = _DRY_RUN_BANKROLL
        log.info("DRY-RUN bankroll: $%.2f (simulado)", bankroll)

    return max(bankroll, 1.0)


def _select_independent_mm_markets(markets: dict[str, Market], cfg) -> dict[str, Market]:
    """Select short-horizon markets for MM without relying on LLM approval."""
    if not cfg.mm_independent_discovery_enabled:
        return {}

    min_minutes = float(cfg.mm_short_horizon_min_minutes)
    max_minutes = float(cfg.mm_short_horizon_max_hours) * 60.0

    shortlisted: list[Market] = []
    for market in markets.values():
        if not market.active or market.closed or market.archived:
            continue
        if market.yes_token is None:
            continue
        if market.yes_price is None or not (0.03 < market.yes_price < 0.97):
            continue
        if market.volume_24hr < cfg.mm_min_market_volume_24h_usd:
            continue
        if market.liquidity < cfg.mm_min_market_liquidity_usd:
            continue

        days_to_end = market.days_to_end
        if days_to_end is None:
            continue
        minutes_to_end = days_to_end * 24.0 * 60.0
        if minutes_to_end < min_minutes or minutes_to_end > max_minutes:
            continue

        shortlisted.append(market)

    shortlisted.sort(
        key=lambda m: (m.volume_24hr, m.liquidity),
        reverse=True,
    )

    limited = shortlisted[: cfg.mm_independent_max_markets]
    return {m.condition_id: m for m in limited}


async def _slow_loop(
    evaluator: LLMEvaluator,
    maker: MarketMaker,
    brain: Brain,
    provider,
    clob: AsyncClobClient,
    ws: WsClient,
    user_ws,          # UserWsClient | None
    cfg,
) -> None:
    """LLM evaluation cycle (every CYCLE_INTERVAL_SECONDS).

    Discovers markets, runs LLM analysis, and feeds approved markets
    into the MarketMaker + Brain for spread capture.
    Also keeps both WebSocket channels (book + user fills) in sync
    with the current approved market set.
    """
    cycle = 0
    while not _shutdown_event.is_set():
        cycle += 1
        log.info("═══ LLM Cycle %d ═══", cycle)

        try:
            bankroll = await _get_bankroll(provider, cfg)
            summary = await evaluator.run_cycle(bankroll=bankroll)
            log.info("LLM cycle summary: %s", summary)

            # Feed approved markets to the market maker and brain
            approved = evaluator.get_approved_markets()
            if cfg.mm_enabled:
                raw_markets = await clob.get_markets(limit=cfg.mm_independent_fetch_limit)
                discovered = _select_independent_mm_markets(
                    {m.condition_id: m for m in raw_markets},
                    cfg,
                )
                mm_pool = dict(approved)
                mm_pool.update(discovered)
                maker.update_active_markets(mm_pool)
                brain.update_approved_markets(mm_pool)

                # Update order-book WS subscriptions (public channel)
                token_ids = [
                    m.yes_token.token_id
                    for m in mm_pool.values()
                    if m.yes_token
                ]
                await ws.subscribe(token_ids)

                # Update user-fill WS subscriptions (authenticated channel)
                if user_ws is not None:
                    condition_ids = list(mm_pool.keys())
                    await user_ws.subscribe(condition_ids)

                log.info(
                    "MM market pool updated: llm=%d short_horizon=%d total=%d "
                    "(book_tokens=%d fill_markets=%d)",
                    len(approved),
                    len(discovered),
                    len(mm_pool),
                    len(token_ids),
                    len(mm_pool) if user_ws else 0,
                )

        except Exception as exc:
            log.exception("Unexpected error in LLM cycle %d: %s", cycle, exc)

        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=cfg.cycle_interval_seconds,
            )
        except asyncio.TimeoutError:
            pass


async def _fast_loop(maker: MarketMaker, db: Database, cfg) -> None:
    """Market-making fast loop — now wakes on signal queue, not a fixed timer.

    Reads ExecutionSignal / CancelSignal objects from the Brain's queue
    and dispatches them immediately.  Falls back to a 1s timeout poll so
    time-based checks (stale orders, trailing exit) still run.
    """
    tick = 0
    while not _shutdown_event.is_set():
        tick += 1

        try:
            summary = await maker.run_tick()

            if any(v > 0 for v in summary.values()):
                daily_pnl = await db.get_daily_mm_pnl()
                log.info(
                    "MM tick %d: %s | daily P&L: %+.4f",
                    tick, summary, daily_pnl,
                )
        except Exception as exc:
            log.error("MM tick %d error: %s", tick, exc)

        # Use a short timeout so the loop stays responsive to shutdown
        # The event-driven part (Brain signals) is handled via the signal_queue
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=cfg.mm_cycle_seconds,
            )
        except asyncio.TimeoutError:
            pass



# ── Health check server ───────────────────────────────────────────────────────

async def _health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Respuestas HTTP muy simples para el Docker healthcheck o balanceadores."""
    try:
        # Read and discard request line + headers to avoid abrupt socket resets
        # on some clients when closing a connection with unread data.
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=0.2)
        except Exception:
            pass

        if _shutdown_event.is_set():
            status = b"503 Service Unavailable"
            body = b"DOWN\n"
        else:
            status = b"200 OK"
            body = b"OK\n"

        headers = (
            b"HTTP/1.1 " + status + b"\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n"
        )
        response = headers + body
        writer.write(response)
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def _run_health_server(port: int) -> None:
    server = await asyncio.start_server(_health_handler, "0.0.0.0", port)
    log.info("Health check server listening on 0.0.0.0:%d", port)
    async with server:
        await _shutdown_event.wait()


# ── DNS pre-check ─────────────────────────────────────────────────────────────

_DNS_HOSTS = ["clob.polymarket.com", "gamma-api.polymarket.com"]

def _check_dns() -> bool:
    """Return True if all required hostnames resolve; log actionable help if not."""
    ok = True
    for host in _DNS_HOSTS:
        try:
            socket.getaddrinfo(host, 443)
        except socket.gaierror:
            ok = False
            log.error(
                "DNS no puede resolver '%s'.\n"
                "  Solución: cambia el DNS de tu Mac a 8.8.8.8 (Google):\n"
                "  Configuración del Sistema → Red → <conexión activa> → Detalles → DNS\n"
                "  Agrega 8.8.8.8 y 1.1.1.1, aplica, y ejecuta en Terminal:\n"
                "    sudo dscacheutil -flushcache && sudo killall -HUP mDNSResponder",
                host,
            )
    return ok


# ── Application bootstrap ─────────────────────────────────────────────────────

async def main() -> None:
    cfg = get_settings()

    _setup_logging(cfg)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Polymarket Intelligence Bot  |  dry_run=%s", cfg.dry_run)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    _install_signal_handlers()

    # ── 0. DNS sanity check ────────────────────────────────────────────────────
    if not _check_dns():
        log.error("Resuelve el problema de DNS antes de continuar. Abortando.")
        return

    # ── 1. Validate required config ────────────────────────────────────────────
    log.info("LLM provider: %s", cfg.llm_provider)

    if cfg.llm_provider == "gemini" and not cfg.gemini_api_key:
        log.error(
            "GEMINI_API_KEY no está configurado. "
            "Obtén tu clave gratuita en https://aistudio.google.com/app/apikey "
            "y agrégala al archivo .env"
        )
        return

    if cfg.llm_provider == "ollama":
        log.info("Usando Ollama local — modelo: %s @ %s", cfg.ollama_model, cfg.ollama_base_url)

    # ── 2. Private key from macOS Keychain ─────────────────────────────────────
    private_key: Optional[str] = None
    if not cfg.dry_run:
        private_key = get_private_key(cfg.keychain_service, cfg.keychain_account)
        if not private_key:
            log.error(
                "No private key found in Keychain.  Run:\n"
                "  python -c 'from src.core.security import store_key; store_key()'"
            )
            return
        log.info("Private key loaded from Keychain (service='%s').", cfg.keychain_service)
    else:
        log.info("[DRY-RUN] Skipping Keychain lookup — no real orders will be placed.")
        # Use a throwaway key for initialising the CLOB client in read-only mode
        private_key = "0x" + "1" * 64

    # ── 3. Database ────────────────────────────────────────────────────────────
    db = Database(cfg.db_path)
    await db.connect()

    # ── 4. Polygon RPC ─────────────────────────────────────────────────────────
    provider: Optional[PolyProvider] = None
    try:
        provider = await PolyProvider.create(cfg.rpc_url_list)
        block = await provider.get_block_number()
        log.info("Polygon block: %d (via %s)", block, provider.active)
    except Exception as exc:
        log.warning("Could not connect to Polygon RPC: %s — balance checks disabled.", exc)
        provider = None

    # ── 5. CLOB client ─────────────────────────────────────────────────────────
    clob = AsyncClobClient(
        host=cfg.polymarket_host,
        private_key=private_key,
        chain_id=cfg.polymarket_chain_id,
        dry_run=cfg.dry_run,   # omite derivación de creds L2 en dry-run
    )
    await clob.initialize()

    # ── 6. RAG engine (embeddings locales, sin API de pago) ───────────────────
    rag = RagEngine(chroma_path=cfg.chroma_path)
    await rag.initialize()
    doc_count = await rag.document_count()
    log.info("RAG engine ready — %d documents in index.", doc_count)

    # ── 7. Strategy engines ─────────────────────────────────────────────────────
    signal_queue = asyncio.Queue(maxsize=500)
    book_cache   = BookCache(max_age_seconds=5.0)
    fill_cache   = FillCache(max_age_seconds=300.0)
    ws_client    = WsClient(book_cache=book_cache)

    # Build L2 API credentials for the user WebSocket (requires live mode)
    user_api_creds = make_api_creds_from_clob(clob)
    user_ws = UserWsClient(
        fill_cache=fill_cache,
        api_creds=user_api_creds,
    ) if (user_api_creds and not cfg.dry_run) else None

    if user_ws is None:
        log.info(
            "UserWsClient: disabled (dry_run=%s, creds=%s) — fill detection falls back to HTTP",
            cfg.dry_run, "missing" if user_api_creds is None else "OK",
        )

    evaluator    = LLMEvaluator(clob=clob, rag=rag, db=db, settings=cfg)
    maker        = MarketMaker(
        clob=clob,
        db=db,
        settings=cfg,
        fill_cache=fill_cache if user_ws else None,
        book_cache=book_cache,
    )
    stat_arber   = WsStatArbEngine(cfg=cfg, clob=clob, ws=ws_client, cache=book_cache, db=db)
    brain        = Brain(
        book_cache=book_cache,
        signal_queue=signal_queue,
        db=db,
        settings=cfg,
    )

    # ── 8. Dual loop ─────────────────────────────────────────────────────────
    if provider is None:
        class _NoProvider:  # type: ignore[no-redef]
            async def get_usdc_balance(self, _: str) -> float:
                return 0.0

        provider = _NoProvider()  # type: ignore[assignment]

    loops = [
        ws_client.stream(),                                          # public WS: order book
        brain.run(),                                                 # event-driven analysis
        _slow_loop(evaluator, maker, brain, provider, clob, ws_client, user_ws, cfg),
        _run_health_server(cfg.health_port),
    ]
    if user_ws is not None:
        loops.append(user_ws.stream())   # authenticated WS: fill/cancel events
    if cfg.mm_enabled:
        loops.append(_fast_loop(maker, db, cfg))
        log.info(
            "Starting: WsClient + Brain + LLM (every %ds) + MM (every %ds)",
            cfg.cycle_interval_seconds, cfg.mm_cycle_seconds,
        )
    else:
        log.info(
            "Starting: WsClient + Brain + LLM (every %ds) — MM disabled",
            cfg.cycle_interval_seconds,
        )

    if cfg.dh15m_enabled:
        await stat_arber.initialize_clusters()
        loops.append(stat_arber.run())
        log.info(
            "WS StatArb Engine enabled (Cross-Market & Spread Arb): sum_target=%.3f",
            cfg.dh15m_sum_target,
        )

    try:
        ws_mode = "WS(book+fills)" if user_ws else "WS(book-only)"
        await notifier.send_message(
            f"Bot Iniciado | Dry Run: {cfg.dry_run} | MM: {cfg.mm_enabled} | {ws_mode}"
        )
        await asyncio.gather(*loops)
    finally:
        log.info("Shutting down…")
        await notifier.send_message("🛑 <b>Polymarket Bot Apagado</b>")
        await ws_client.stop()
        if user_ws is not None:
            await user_ws.stop()
        await brain.stop()
        await maker.cancel_all()
        await clob.close()
        await db.close()
        if hasattr(provider, "close"):
            await provider.close()  # type: ignore[union-attr]
        log.info("Bot stopped cleanly.")



if __name__ == "__main__":
    asyncio.run(main())
