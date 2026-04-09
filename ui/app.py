"""
Polymarket Intelligence — main Textual TUI application.

Navigation: arrow keys / mouse to move, Tab/Shift-Tab to switch tabs.
Press 'q' to quit, 'r' to refresh the current tab.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from analytics.arbitrage import find_arbitrage_opportunities
from analytics.risk import rank_by_risk, score_market
from analytics.whale import detect_whale_trades, whale_sentiment
from api.kalshi import kalshi_client
from api.polymarket import poly_client
from config import (
    APP_TITLE,
    APP_VERSION,
    REFRESH_ALERTS,
    REFRESH_ARBITRAGE,
    REFRESH_MARKETS,
    REFRESH_ORDERBOOK,
    REFRESH_PORTFOLIO,
    REFRESH_RISK,
    REFRESH_WHALES,
)
from database.db import db
from ui.charts import (
    fmt_price,
    fmt_pnl,
    fmt_usd,
    order_book_chart,
    risk_gauge,
    sentiment_bar,
    sparkline,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

APP_CSS = """
Screen {
    background: $background;
}

/* ── Toolbar row at the top of every tab ─────────────────────── */
.toolbar {
    height: 3;
    background: $panel;
    padding: 0 1;
    align: left middle;
}

.toolbar Label {
    color: $accent;
    text-style: bold;
    margin-right: 2;
    width: auto;
}

.toolbar Input {
    width: 28;
    margin-right: 1;
}

/* ── Status bar at the bottom of every tab ───────────────────── */
.statusbar {
    height: 1;
    background: $primary-background;
    padding: 0 1;
    color: $text-muted;
}

/* ── Tables ──────────────────────────────────────────────────── */
DataTable {
    height: 1fr;
}

/* ── Scrollable Rich-text panels (order book, etc.) ──────────── */
.panel {
    height: 1fr;
    overflow-y: auto;
    padding: 1 2;
    background: $surface;
    border: round $primary;
}

/* ── Modal dialogs ───────────────────────────────────────────── */
ModalScreen {
    align: center middle;
}

.dialog {
    width: 64;
    height: auto;
    max-height: 90vh;
    background: $surface;
    border: double $accent;
    padding: 1 2;
}

.dialog Label {
    margin-bottom: 1;
}

.dialog Input {
    margin-bottom: 1;
    width: 100%;
}

.dialog Select {
    margin-bottom: 1;
    width: 100%;
}

.dialog-title {
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}

.dialog-buttons {
    height: 3;
    align: right middle;
    margin-top: 1;
}

.dialog-buttons Button {
    margin-left: 1;
}

/* ── Inline stat blocks ──────────────────────────────────────── */
.stat-box {
    width: 1fr;
    height: 5;
    background: $panel;
    border: round $primary;
    padding: 0 1;
    align: center middle;
    margin: 0 1;
}

.stat-value {
    text-style: bold;
    color: $accent;
}

/* ── Section headers ─────────────────────────────────────────── */
.section-header {
    background: $primary-background;
    padding: 0 1;
    color: $text-muted;
    height: 1;
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _yes_price(market: dict) -> float | None:
    try:
        prices   = market.get("outcomePrices") or []
        outcomes = market.get("outcomes") or []
        if prices and outcomes:
            for i, o in enumerate(outcomes):
                if str(o).lower() == "yes" and i < len(prices):
                    return float(prices[i])
        if prices:
            return float(prices[0])
    except (ValueError, TypeError):
        pass
    return None


def _end_date(market: dict) -> str:
    raw = market.get("endDateIso") or market.get("endDate") or ""
    return raw[:10] if raw else "—"


def _token_ids(market: dict) -> list[str]:
    return list(market.get("clobTokenIds") or [])


# ═══════════════════════════════════════════════════════════════════════════════
# Modal Screens
# ═══════════════════════════════════════════════════════════════════════════════

class BookmarkModal(ModalScreen[dict | None]):
    """Add or view a bookmark."""

    def __init__(self, market: dict) -> None:
        super().__init__()
        self._market = market

    def compose(self) -> ComposeResult:
        q = self._market.get("question", "")[:70]
        with Vertical(classes="dialog"):
            yield Label("Bookmark Market", classes="dialog-title")
            yield Label(f"[dim]{q}[/dim]")
            yield Label("Notes (optional):")
            yield Input(id="notes", placeholder="Your notes about this market…")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            notes = self.query_one("#notes", Input).value.strip()
            self.dismiss({"notes": notes})
        else:
            self.dismiss(None)


class AlertModal(ModalScreen[dict | None]):
    """Create a price threshold alert."""

    def __init__(self, market: dict) -> None:
        super().__init__()
        self._market = market

    def compose(self) -> ComposeResult:
        q = self._market.get("question", "")[:70]
        with Vertical(classes="dialog"):
            yield Label("New Price Alert", classes="dialog-title")
            yield Label(f"[dim]{q}[/dim]")
            yield Label("Outcome:")
            yield Select(
                [("YES", "YES"), ("NO", "NO")],
                id="outcome",
                value="YES",
            )
            yield Label("Trigger when price is:")
            yield Select(
                [("above", "above"), ("below", "below")],
                id="condition",
                value="above",
            )
            yield Label("Target price (0.00 – 1.00):")
            yield Input(id="target", placeholder="e.g. 0.75")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Create Alert", id="save", variant="success")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        try:
            outcome    = str(self.query_one("#outcome", Select).value)
            condition  = str(self.query_one("#condition", Select).value)
            target_str = self.query_one("#target", Input).value.strip()
            target     = float(target_str)
            if not (0 < target < 1):
                raise ValueError
            self.dismiss({"outcome": outcome, "condition": condition, "target": target})
        except (ValueError, TypeError):
            self.app.notify("Invalid price. Enter a number between 0 and 1.", severity="error")


class JournalModal(ModalScreen[dict | None]):
    """Record a journal observation."""

    def __init__(self, market: dict | None = None) -> None:
        super().__init__()
        self._market = market or {}

    def compose(self) -> ComposeResult:
        q = self._market.get("question", "")
        with Vertical(classes="dialog"):
            yield Label("New Journal Entry", classes="dialog-title")
            yield Label("Market question:")
            yield Input(id="question", value=q[:100], placeholder="Market question…")
            yield Label("Trade type:")
            yield Select([("BUY", "BUY"), ("SELL", "SELL")], id="trade_type", value="BUY")
            yield Label("Outcome:")
            yield Select([("YES", "YES"), ("NO", "NO")], id="outcome", value="YES")
            yield Label("Price (0–1):")
            yield Input(id="price", placeholder="e.g. 0.65")
            yield Label("Size (USD):")
            yield Input(id="size", placeholder="e.g. 100")
            yield Label("Rationale / notes:")
            yield Input(id="rationale", placeholder="Why are you making this trade?")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        try:
            question   = self.query_one("#question",   Input).value.strip()
            trade_type = str(self.query_one("#trade_type", Select).value)
            outcome    = str(self.query_one("#outcome",    Select).value)
            price      = float(self.query_one("#price", Input).value)
            size       = float(self.query_one("#size",  Input).value)
            rationale  = self.query_one("#rationale", Input).value.strip()
            if not question or not (0 < price < 1) or size <= 0:
                raise ValueError
            self.dismiss({
                "question": question,
                "trade_type": trade_type,
                "outcome": outcome,
                "price": price,
                "size": size,
                "rationale": rationale,
                "market_id": self._market.get("id"),
            })
        except (ValueError, TypeError):
            self.app.notify("Please fill all fields correctly.", severity="error")


class ResolveJournalModal(ModalScreen[dict | None]):
    """Mark a journal entry as WIN / LOSS and record P&L."""

    def __init__(self, entry_id: int) -> None:
        super().__init__()
        self._entry_id = entry_id

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Resolve Journal Entry", classes="dialog-title")
            yield Label("Result:")
            yield Select([("WIN", "WIN"), ("LOSS", "LOSS"), ("PENDING", "PENDING")],
                         id="result", value="WIN")
            yield Label("P&L (USD, can be negative):")
            yield Input(id="pnl", placeholder="e.g. 45.50 or -20.00")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        try:
            result = str(self.query_one("#result", Select).value)
            pnl_s  = self.query_one("#pnl", Input).value.strip()
            pnl    = float(pnl_s) if pnl_s else None
            self.dismiss({"result": result, "pnl": pnl, "entry_id": self._entry_id})
        except (ValueError, TypeError):
            self.app.notify("Invalid P&L value.", severity="error")


class WalletModal(ModalScreen[dict | None]):
    """Save a wallet address."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Save Wallet Address", classes="dialog-title")
            yield Label("[dim]View-only — no keys are stored[/dim]")
            yield Label("Address (0x…):")
            yield Input(id="address", placeholder="0x…")
            yield Label("Label (optional):")
            yield Input(id="label", placeholder="e.g. My main wallet")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        address = self.query_one("#address", Input).value.strip()
        label   = self.query_one("#label",   Input).value.strip()
        if not address.startswith("0x") or len(address) < 10:
            self.app.notify("Enter a valid 0x wallet address.", severity="error")
            return
        self.dismiss({"address": address, "label": label})


# ═══════════════════════════════════════════════════════════════════════════════
# Tab Widgets
# ═══════════════════════════════════════════════════════════════════════════════

class MarketsTab(Container):
    """Live market feed sorted by 24 h volume."""

    _markets: reactive[list] = reactive([], recompose=False)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Active Markets")
            yield Input(placeholder="Search…", id="mkt-search")
            yield Button("↻ Refresh", id="mkt-refresh", variant="primary")
        table = DataTable(id="mkt-table", zebra_stripes=True, cursor_type="row")
        table.add_columns(
            "Question", "YES", "NO", "24 h Vol", "Liquidity", "Ends", "★"
        )
        yield table
        yield Label("", id="mkt-status", classes="statusbar")

    def on_mount(self) -> None:
        self.load_markets()

    # ── Data loading ──────────────────────────────────────────────────────────

    @work(exclusive=True)
    async def load_markets(self, query: str = "") -> None:
        status = self.query_one("#mkt-status", Label)
        status.update("[yellow]Loading markets…[/yellow]")

        markets = (
            await poly_client.search_markets(query, limit=80)
            if query
            else await poly_client.get_markets(limit=100)
        )

        table = self.query_one("#mkt-table", DataTable)
        table.clear()

        bm_ids = {b["market_id"] for b in db.get_bookmarks()}

        for m in markets:
            yp   = _yes_price(m)
            np_  = round(1 - yp, 3) if yp is not None else None
            star = "★" if m.get("id") in bm_ids else " "
            table.add_row(
                m.get("question", "")[:62],
                f"{yp:.3f}"  if yp  is not None else "—",
                f"{np_:.3f}" if np_ is not None else "—",
                fmt_usd(float(m.get("volume24hr", 0) or 0)),
                fmt_usd(float(m.get("liquidityNum", 0) or m.get("liquidity", 0) or 0)),
                _end_date(m),
                star,
                key=m.get("id", ""),
            )

        self._markets = markets
        status.update(f"[green]{len(markets)} markets loaded — {_ts()}[/green]")

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mkt-refresh":
            q = self.query_one("#mkt-search", Input).value.strip()
            self.load_markets(q)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "mkt-search":
            self.load_markets(event.value.strip())

    def action_bookmark(self) -> None:
        market = self._selected_market()
        if not market:
            return
        self.app.push_screen(BookmarkModal(market), self._on_bookmark)

    def action_alert(self) -> None:
        market = self._selected_market()
        if not market:
            return
        self.app.push_screen(AlertModal(market), self._on_alert)

    def action_journal(self) -> None:
        market = self._selected_market()
        self.app.push_screen(JournalModal(market or {}), self._on_journal)

    def action_orderbook(self) -> None:
        market = self._selected_market()
        if not market:
            self.app.notify("Select a market first.", severity="warning")
            return
        token_ids = _token_ids(market)
        if not token_ids:
            self.app.notify("No CLOB token available for this market.", severity="warning")
            return
        # Switch to the order-book tab and pre-populate the token field
        self.app.query_one("#tab-orderbook Input", Input).value = token_ids[0]
        self.app.query_one(TabbedContent).active = "orderbook"
        self.app.query_one(OrderBookTab).load_book(token_ids[0], market.get("question", ""))

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_bookmark(self, result: dict | None) -> None:
        market = self._selected_market()
        if result is None or not market:
            return
        ok = db.add_bookmark(
            market.get("id", ""),
            market.get("question", ""),
            token_id=(_token_ids(market) or [None])[0],
            notes=result.get("notes"),
        )
        msg = "Bookmarked!" if ok else "Already bookmarked."
        self.app.notify(msg, severity="information")
        self.load_markets(self.query_one("#mkt-search", Input).value.strip())

    def _on_alert(self, result: dict | None) -> None:
        market = self._selected_market()
        if result is None or not market:
            return
        token_ids = _token_ids(market)
        db.add_alert(
            market_id=market.get("id", ""),
            question=market.get("question", ""),
            token_id=(token_ids[0] if token_ids else ""),
            outcome=result["outcome"],
            condition=result["condition"],
            target_price=result["target"],
        )
        self.app.notify(
            f"Alert set: {result['outcome']} {result['condition']} {result['target']:.3f}",
            severity="information",
        )

    def _on_journal(self, result: dict | None) -> None:
        if not result:
            return
        db.add_journal_entry(
            question=result["question"],
            trade_type=result["trade_type"],
            outcome=result["outcome"],
            price=result["price"],
            size_usd=result["size"],
            rationale=result["rationale"],
            market_id=result.get("market_id"),
        )
        self.app.notify("Journal entry saved.", severity="information")

    def _selected_market(self) -> dict | None:
        table = self.query_one("#mkt-table", DataTable)
        if table.cursor_row < 0 or not self._markets:
            return None
        try:
            return self._markets[table.cursor_row]
        except IndexError:
            return None


# ─────────────────────────────────────────────────────────────────────────────

class WhalesTab(Container):
    """Detect and display large trades across top markets."""

    _all_trades: reactive[list] = reactive([], recompose=False)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Whale Activity  🐋")
            yield Label("Min $ threshold:")
            yield Input(id="whale-thresh", value="5000", placeholder="e.g. 5000")
            yield Button("↻ Refresh", id="whale-refresh", variant="primary")
        table = DataTable(id="whale-table", zebra_stripes=True, cursor_type="row")
        table.add_columns("Market", "Side", "Outcome", "Price", "Size", "Value $", "Maker", "Time")
        yield table
        yield Label("", id="whale-status", classes="statusbar")

    def on_mount(self) -> None:
        self.load_whales()

    @work(exclusive=True)
    async def load_whales(self) -> None:
        status = self.query_one("#whale-status", Label)
        status.update("[yellow]Fetching top markets for whale scan…[/yellow]")

        try:
            thresh_str = self.query_one("#whale-thresh", Input).value.strip()
            thresh = float(thresh_str) if thresh_str else 5000.0
        except ValueError:
            thresh = 5000.0

        markets = await poly_client.get_markets(limit=20)
        status.update("[yellow]Fetching recent trades…[/yellow]")

        all_trades = await poly_client.get_recent_trades_bulk(markets, per_market=40)
        whales     = detect_whale_trades(all_trades, threshold_usd=thresh)

        table = self.query_one("#whale-table", DataTable)
        table.clear()

        for w in whales[:200]:
            ts  = (w.get("timestamp") or "")[:16].replace("T", " ")
            val = w["value_usd"]
            color = "bold red" if val >= 50_000 else "yellow" if val >= 15_000 else "white"
            table.add_row(
                w["question"][:50],
                w["side"],
                w["outcome"],
                f"{w['price']:.3f}",
                fmt_usd(w["size"]),
                f"[{color}]{fmt_usd(val)}[/{color}]",
                w["maker"],
                ts,
            )

        sent = whale_sentiment(all_trades)
        bar  = sentiment_bar(sent["buy_pct"])
        status.update(
            f"[green]{len(whales)} whale trades — {sent['sentiment']} {bar} — {_ts()}[/green]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "whale-refresh":
            self.load_whales()


# ─────────────────────────────────────────────────────────────────────────────

class ArbitrageTab(Container):
    """Cross-platform price comparison: Polymarket vs Kalshi."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Arbitrage Scanner  ⚡ (Polymarket ↔ Kalshi)")
            yield Button("↻ Scan", id="arb-refresh", variant="primary")
        table = DataTable(id="arb-table", zebra_stripes=True, cursor_type="row")
        table.add_columns(
            "Polymarket Question",
            "Kalshi Title",
            "Sim %",
            "PM YES",
            "KS YES",
            "Spread",
            "Spread %",
            "Buy on",
        )
        yield table
        yield Static(
            "[dim]Spread = |Polymarket YES − Kalshi YES|.  "
            "Min similarity 45 %.  Min spread 2 %.[/dim]",
            classes="statusbar",
        )
        yield Label("", id="arb-status", classes="statusbar")

    def on_mount(self) -> None:
        self.scan()

    @work(exclusive=True)
    async def scan(self) -> None:
        status = self.query_one("#arb-status", Label)
        status.update("[yellow]Fetching markets from both platforms…[/yellow]")

        poly_markets, kalshi_markets = await asyncio.gather(
            poly_client.get_markets(limit=100),
            kalshi_client.get_markets(limit=200),
        )

        opps = find_arbitrage_opportunities(poly_markets, kalshi_markets)

        table = self.query_one("#arb-table", DataTable)
        table.clear()

        for o in opps:
            spread_pct = o["spread_pct"]
            color = "bold red" if spread_pct >= 8 else "yellow" if spread_pct >= 4 else "white"
            table.add_row(
                o["poly_question"][:45],
                o["kalshi_title"][:40],
                f"{o['similarity_pct']:.0f}%",
                f"{o['poly_yes']:.3f}",
                f"{o['kalshi_yes']:.3f}",
                f"{o['spread']:+.4f}",
                f"[{color}]{spread_pct:.2f}%[/{color}]",
                o["buy_on"],
            )

        msg = (
            f"[green]{len(opps)} opportunities found — "
            f"{len(poly_markets)} PM markets, {len(kalshi_markets)} Kalshi markets — {_ts()}[/green]"
            if opps
            else f"[dim]No opportunities above threshold. {_ts()}[/dim]"
        )
        status.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "arb-refresh":
            self.scan()


# ─────────────────────────────────────────────────────────────────────────────

class OrderBookTab(Container):
    """Real-time order-book depth chart for any market token."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar", id="tab-orderbook"):
            yield Label("Order Book Depth  📊")
            yield Input(id="ob-token", placeholder="Paste YES token ID…")
            yield Input(id="ob-label", placeholder="Market name (optional)")
            yield Button("Load", id="ob-load", variant="primary")
            yield Button("↻", id="ob-refresh", variant="default")
        with ScrollableContainer(classes="panel"):
            yield Static("", id="ob-chart")
        yield Label("", id="ob-status", classes="statusbar")

    def load_book(self, token_id: str, label: str = "") -> None:
        if label:
            self.query_one("#ob-label", Input).value = label[:60]
        self.query_one("#ob-token", Input).value = token_id
        self._fetch_book(token_id)

    @work(exclusive=True)
    async def _fetch_book(self, token_id: str) -> None:
        status = self.query_one("#ob-status", Label)
        chart  = self.query_one("#ob-chart",  Static)
        status.update("[yellow]Fetching order book…[/yellow]")

        book = await poly_client.get_order_book(token_id)

        if not book:
            chart.update("[red]Could not load order book. Check the token ID.[/red]")
            status.update("[red]Error loading order book.[/red]")
            return

        bids_raw = book.get("bids") or []
        asks_raw = book.get("asks") or []

        def parse(levels: list[dict]) -> list[tuple[float, float]]:
            out = []
            for lv in levels:
                try:
                    p = float(lv.get("price", 0))
                    s = float(lv.get("size",  0))
                    if p > 0 and s > 0:
                        out.append((p, s))
                except (ValueError, TypeError):
                    continue
            return out

        bids = sorted(parse(bids_raw), key=lambda x: x[0], reverse=True)
        asks = sorted(parse(asks_raw), key=lambda x: x[0])

        label = self.query_one("#ob-label", Input).value or token_id[:20]
        header = f"\n[bold white]{label}[/bold white]\n\n"
        chart.update(header + order_book_chart(bids, asks))

        # Summary stats
        total_bid = sum(s for _, s in bids)
        total_ask = sum(s for _, s in asks)
        status.update(
            f"[green]Bid depth: {fmt_usd(total_bid)}  ·  "
            f"Ask depth: {fmt_usd(total_ask)}  ·  {_ts()}[/green]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        token = self.query_one("#ob-token", Input).value.strip()
        if not token:
            self.app.notify("Enter a token ID.", severity="warning")
            return
        if event.button.id in ("ob-load", "ob-refresh"):
            self._fetch_book(token)


# ─────────────────────────────────────────────────────────────────────────────

class RiskTab(Container):
    """Risk-score leaderboard across active markets."""

    _markets: reactive[list] = reactive([], recompose=False)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Risk Analytics  🎯")
            yield Button("↻ Refresh", id="risk-refresh", variant="primary")
        table = DataTable(id="risk-table", zebra_stripes=True, cursor_type="row")
        table.add_columns(
            "Market", "Score", "Level",
            "Liquidity", "Activity", "Time", "Certainty", "Age",
            "Liq $", "Vol/24 h",
        )
        yield table
        yield Label("", id="risk-status", classes="statusbar")

    def on_mount(self) -> None:
        self.load_risk()

    @work(exclusive=True)
    async def load_risk(self) -> None:
        status = self.query_one("#risk-status", Label)
        status.update("[yellow]Scoring markets…[/yellow]")

        markets = await poly_client.get_markets(limit=100)
        ranked  = rank_by_risk(markets)

        table = self.query_one("#risk-table", DataTable)
        table.clear()

        self._markets = ranked

        for m in ranked:
            r   = m["_risk"]
            bd  = r["breakdown"]
            lvl = r["risk_level"]
            clr = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(lvl, "white")
            table.add_row(
                m.get("question", "")[:55],
                risk_gauge(r["composite_score"], width=12),
                f"[{clr}]{lvl}[/{clr}]",
                f"{bd['liquidity']:.0f}",
                f"{bd['activity']:.0f}",
                f"{bd['time']:.0f}",
                f"{bd['certainty']:.0f}",
                f"{bd['age']:.0f}",
                fmt_usd(r["liquidity_usd"]),
                fmt_usd(r["volume_24h_usd"]),
                key=m.get("id", ""),
            )

        status.update(f"[green]{len(ranked)} markets scored — {_ts()}[/green]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "risk-refresh":
            self.load_risk()


# ─────────────────────────────────────────────────────────────────────────────

class PortfolioTab(Container):
    """View-only wallet portfolio: positions and P&L."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Portfolio  💼  [dim](view-only)[/dim]")
            yield Input(id="wallet-addr", placeholder="0x… wallet address")
            yield Button("Load", id="port-load", variant="primary")
            yield Button("Save", id="port-save", variant="default")
        table = DataTable(id="port-table", zebra_stripes=True, cursor_type="row")
        table.add_columns(
            "Market", "Outcome", "Shares", "Avg Entry", "Cur Price", "Value $", "P&L"
        )
        yield table
        with Horizontal(id="port-stats"):
            yield Static("", id="port-stat-value",  classes="stat-box")
            yield Static("", id="port-stat-pnl",    classes="stat-box")
            yield Static("", id="port-stat-count",  classes="stat-box")
        yield Label("", id="port-status", classes="statusbar")

    def on_mount(self) -> None:
        wallets = db.get_wallets()
        if wallets:
            self.query_one("#wallet-addr", Input).value = wallets[0]["address"]
            self._load_wallet(wallets[0]["address"])

    @work(exclusive=True)
    async def _load_wallet(self, address: str) -> None:
        status = self.query_one("#port-status", Label)
        status.update("[yellow]Loading positions…[/yellow]")

        positions, port_value = await asyncio.gather(
            poly_client.get_positions(address),
            poly_client.get_value(address),
        )

        table = self.query_one("#port-table", DataTable)
        table.clear()

        total_value = 0.0
        total_pnl   = 0.0

        for p in positions:
            try:
                size    = float(p.get("size",         0) or 0)
                avg_in  = float(p.get("avgPrice",     0) or p.get("initialValue", 0) or 0)
                cur_p   = float(p.get("curPrice",     0) or p.get("currentPrice", 0) or 0)
                value   = float(p.get("value",        0) or size * cur_p)
                cost    = float(p.get("initialValue", 0) or size * avg_in)
                pnl     = value - cost
                total_value += value
                total_pnl   += pnl

                outcome = p.get("outcome", "").upper() or "YES"
                clr     = "green" if pnl >= 0 else "red"
                table.add_row(
                    (p.get("title") or p.get("question") or "")[:55],
                    outcome,
                    f"{size:,.2f}",
                    f"{avg_in:.3f}",
                    f"{cur_p:.3f}",
                    fmt_usd(value),
                    f"[{clr}]{fmt_pnl(pnl)}[/{clr}]",
                )
            except (ValueError, TypeError):
                continue

        # Summary stats
        tv = port_value.get("portfolioValue") if port_value else None
        v_display = fmt_usd(float(tv) if tv else total_value)

        self.query_one("#port-stat-value", Static).update(
            f"[dim]Portfolio Value[/dim]\n[bold accent]{v_display}[/bold accent]"
        )
        pnl_color = "green" if total_pnl >= 0 else "red"
        self.query_one("#port-stat-pnl", Static).update(
            f"[dim]Unrealised P&L[/dim]\n[bold {pnl_color}]{fmt_pnl(total_pnl)}[/bold {pnl_color}]"
        )
        self.query_one("#port-stat-count", Static).update(
            f"[dim]Open Positions[/dim]\n[bold]{len(positions)}[/bold]"
        )

        msg = (
            f"[green]{len(positions)} positions — {address[:10]}… — {_ts()}[/green]"
            if positions
            else f"[dim]No open positions found for {address[:20]}…[/dim]"
        )
        status.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        addr = self.query_one("#wallet-addr", Input).value.strip()
        if event.button.id == "port-load":
            if not addr:
                self.app.notify("Enter a wallet address.", severity="warning")
                return
            self._load_wallet(addr)
        elif event.button.id == "port-save":
            self.app.push_screen(WalletModal(), self._on_wallet_saved)

    def _on_wallet_saved(self, result: dict | None) -> None:
        if not result:
            return
        db.add_wallet(result["address"], result["label"])
        self.app.notify(f"Saved: {result['address'][:20]}…", severity="information")


# ─────────────────────────────────────────────────────────────────────────────

class BookmarksTab(Container):
    """Browse and manage saved market bookmarks."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Bookmarks  ★")
            yield Button("Open Book", id="bm-open",   variant="primary")
            yield Button("Delete",    id="bm-delete",  variant="error")
            yield Button("↻ Refresh", id="bm-refresh", variant="default")
        table = DataTable(id="bm-table", zebra_stripes=True, cursor_type="row")
        table.add_columns("Market Question", "Token ID", "Notes", "Saved")
        yield table
        yield Label("", id="bm-status", classes="statusbar")

    def on_mount(self) -> None:
        self._load()

    def _load(self) -> None:
        bookmarks = db.get_bookmarks()
        table = self.query_one("#bm-table", DataTable)
        table.clear()
        for b in bookmarks:
            table.add_row(
                b["question"][:60],
                (b.get("token_id") or "—")[:20],
                (b.get("notes") or "—")[:40],
                (b.get("created_at") or "")[:10],
                key=b["market_id"],
            )
        self.query_one("#bm-status", Label).update(
            f"[green]{len(bookmarks)} bookmarks[/green]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        table = self.query_one("#bm-table", DataTable)
        if event.button.id == "bm-refresh":
            self._load()
            return

        row_key = _selected_key(table)
        if row_key is None:
            self.app.notify("Select a bookmark first.", severity="warning")
            return

        if event.button.id == "bm-delete":
            db.remove_bookmark(row_key)
            self.app.notify("Bookmark removed.", severity="information")
            self._load()
        elif event.button.id == "bm-open":
            bookmarks = {b["market_id"]: b for b in db.get_bookmarks()}
            bm = bookmarks.get(row_key)
            if bm and bm.get("token_id"):
                ob_tab = self.app.query_one(OrderBookTab)
                ob_tab.load_book(bm["token_id"], bm["question"])
                self.app.query_one(TabbedContent).active = "orderbook"
            else:
                self.app.notify("No CLOB token saved for this bookmark.", severity="warning")


# ─────────────────────────────────────────────────────────────────────────────

class AlertsTab(Container):
    """Price alerts — checked every 30 s against live market data."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Price Alerts  🔔")
            yield Button("Delete",   id="al-delete",  variant="error")
            yield Button("↻ Refresh", id="al-refresh", variant="default")
        table = DataTable(id="al-table", zebra_stripes=True, cursor_type="row")
        table.add_columns(
            "Market", "Outcome", "Condition", "Target",
            "Triggered", "Triggered At", "Created"
        )
        yield table
        yield Label("", id="al-status", classes="statusbar")

    def on_mount(self) -> None:
        self._load()

    def _load(self) -> None:
        alerts = db.get_alerts()
        table  = self.query_one("#al-table", DataTable)
        table.clear()
        for a in alerts:
            triggered = "✓" if a["triggered"] else "○"
            trig_at   = (a.get("triggered_at") or "—")[:16]
            clr       = "dim" if a["triggered"] else "white"
            table.add_row(
                f"[{clr}]{a['question'][:55]}[/{clr}]",
                a["outcome"],
                a["condition"].upper(),
                f"{a['target_price']:.3f}",
                triggered,
                trig_at,
                (a.get("created_at") or "")[:10],
                key=str(a["id"]),
            )
        active = sum(1 for a in alerts if not a["triggered"])
        self.query_one("#al-status", Label).update(
            f"[green]{len(alerts)} alerts ({active} active)[/green]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "al-refresh":
            self._load()
            return
        table   = self.query_one("#al-table", DataTable)
        row_key = _selected_key(table)
        if row_key is None:
            self.app.notify("Select an alert first.", severity="warning")
            return
        if event.button.id == "al-delete":
            db.delete_alert(int(row_key))
            self.app.notify("Alert deleted.", severity="information")
            self._load()

    async def check_alerts(self) -> None:
        """Fetch current prices and trigger matching alerts. Called by the app timer."""
        active_alerts = db.get_alerts(active_only=True)
        if not active_alerts:
            return

        token_ids = list({a["token_id"] for a in active_alerts if a.get("token_id")})
        for token_id in token_ids:
            book = await poly_client.get_order_book(token_id)
            if not book:
                continue

            try:
                bids_raw = book.get("bids") or []
                asks_raw = book.get("asks") or []
                best_bid = float(bids_raw[0]["price"]) if bids_raw else None
                best_ask = float(asks_raw[0]["price"]) if asks_raw else None
                if best_bid is None or best_ask is None:
                    continue
                mid = (best_bid + best_ask) / 2
            except (ValueError, IndexError, TypeError):
                continue

            for alert in active_alerts:
                if alert.get("token_id") != token_id:
                    continue
                target    = alert["target_price"]
                condition = alert["condition"].lower()
                triggered = (
                    (condition == "above" and mid >= target) or
                    (condition == "below" and mid <= target)
                )
                if triggered:
                    db.trigger_alert(alert["id"])
                    outcome = alert.get("outcome", "YES")
                    self.app.notify(
                        f"🔔 Alert! {outcome} {condition} {target:.3f} "
                        f"— {alert['question'][:40]}",
                        severity="warning",
                        timeout=8,
                    )

        self._load()


# ─────────────────────────────────────────────────────────────────────────────

class JournalTab(Container):
    """Personal trade journal with running P&L summary."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Trade Journal  📓")
            yield Button("+ New",   id="jnl-add",     variant="success")
            yield Button("Resolve", id="jnl-resolve",  variant="primary")
            yield Button("Delete",  id="jnl-delete",   variant="error")
            yield Button("↻",       id="jnl-refresh",  variant="default")
        table = DataTable(id="jnl-table", zebra_stripes=True, cursor_type="row")
        table.add_columns(
            "Market", "Type", "Out", "Price", "Size $", "Result", "P&L", "Date"
        )
        yield table
        with Horizontal(id="jnl-stats"):
            yield Static("", id="jnl-total",   classes="stat-box")
            yield Static("", id="jnl-winrate", classes="stat-box")
            yield Static("", id="jnl-pnl",     classes="stat-box")
        yield Label("", id="jnl-status", classes="statusbar")

    def on_mount(self) -> None:
        self._load()

    def _load(self) -> None:
        entries = db.get_journal_entries()
        table   = self.query_one("#jnl-table", DataTable)
        table.clear()

        for e in entries:
            res   = (e.get("result") or "PENDING").upper()
            clr   = {"WIN": "green", "LOSS": "red", "PENDING": "dim"}.get(res, "white")
            table.add_row(
                e["question"][:50],
                e["trade_type"],
                e["outcome"],
                f"{e['price']:.3f}",
                fmt_usd(e["size_usd"]),
                f"[{clr}]{res}[/{clr}]",
                fmt_pnl(e.get("pnl")),
                (e.get("created_at") or "")[:10],
                key=str(e["id"]),
            )

        stats = db.get_journal_stats()
        total   = int(stats.get("total",  0) or 0)
        wins    = int(stats.get("wins",   0) or 0)
        losses  = int(stats.get("losses", 0) or 0)
        pnl_sum = float(stats.get("total_pnl", 0) or 0)
        wr      = wins / max(wins + losses, 1) * 100

        self.query_one("#jnl-total",   Static).update(
            f"[dim]Entries[/dim]\n[bold]{total}[/bold]"
        )
        wr_clr = "green" if wr >= 50 else "red"
        self.query_one("#jnl-winrate", Static).update(
            f"[dim]Win Rate[/dim]\n[bold {wr_clr}]{wr:.0f}%[/bold {wr_clr}]"
        )
        p_clr = "green" if pnl_sum >= 0 else "red"
        self.query_one("#jnl-pnl",     Static).update(
            f"[dim]Total P&L[/dim]\n[bold {p_clr}]{fmt_pnl(pnl_sum)}[/bold {p_clr}]"
        )

        self.query_one("#jnl-status", Label).update(
            f"[green]{total} entries — {wins}W / {losses}L — {_ts()}[/green]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "jnl-refresh":
            self._load()
            return

        if event.button.id == "jnl-add":
            self.app.push_screen(JournalModal(), self._on_add)
            return

        table   = self.query_one("#jnl-table", DataTable)
        row_key = _selected_key(table)
        if row_key is None:
            self.app.notify("Select an entry first.", severity="warning")
            return

        if event.button.id == "jnl-resolve":
            self.app.push_screen(ResolveJournalModal(int(row_key)), self._on_resolve)
        elif event.button.id == "jnl-delete":
            db.delete_journal_entry(int(row_key))
            self.app.notify("Entry deleted.", severity="information")
            self._load()

    def _on_add(self, result: dict | None) -> None:
        if not result:
            return
        db.add_journal_entry(
            question=result["question"],
            trade_type=result["trade_type"],
            outcome=result["outcome"],
            price=result["price"],
            size_usd=result["size"],
            rationale=result["rationale"],
            market_id=result.get("market_id"),
        )
        self.app.notify("Journal entry saved.", severity="information")
        self._load()

    def _on_resolve(self, result: dict | None) -> None:
        if not result:
            return
        db.resolve_journal_entry(result["entry_id"], result["result"], result.get("pnl"))
        self.app.notify(f"Entry resolved: {result['result']}", severity="information")
        self._load()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class PolymarketApp(App):
    TITLE   = f"{APP_TITLE}  v{APP_VERSION}"
    CSS     = APP_CSS

    BINDINGS = [
        Binding("q",     "quit",            "Quit"),
        Binding("r",     "refresh",         "Refresh"),
        Binding("b",     "bookmark",        "Bookmark"),
        Binding("a",     "alert",           "Alert"),
        Binding("j",     "journal",         "Journal"),
        Binding("o",     "orderbook",       "Order Book"),
        Binding("d",     "delete_selected", "Delete"),
        Binding("ctrl+c","quit",            "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="markets"):
            with TabPane("📊 Markets",    id="markets"):
                yield MarketsTab(id="tab-markets")
            with TabPane("🐋 Whales",     id="whales"):
                yield WhalesTab(id="tab-whales")
            with TabPane("⚡ Arbitrage",  id="arbitrage"):
                yield ArbitrageTab(id="tab-arbitrage")
            with TabPane("📖 Order Book", id="orderbook"):
                yield OrderBookTab(id="tab-orderbook")
            with TabPane("🎯 Risk",       id="risk"):
                yield RiskTab(id="tab-risk")
            with TabPane("💼 Portfolio",  id="portfolio"):
                yield PortfolioTab(id="tab-portfolio")
            with TabPane("★ Bookmarks",  id="bookmarks"):
                yield BookmarksTab(id="tab-bookmarks")
            with TabPane("🔔 Alerts",    id="alerts"):
                yield AlertsTab(id="tab-alerts")
            with TabPane("📓 Journal",   id="journal"):
                yield JournalTab(id="tab-journal")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(REFRESH_MARKETS,   self._auto_refresh_markets)
        self.set_interval(REFRESH_WHALES,    self._auto_refresh_whales)
        self.set_interval(REFRESH_ARBITRAGE, self._auto_refresh_arb)
        self.set_interval(REFRESH_RISK,      self._auto_refresh_risk)
        self.set_interval(REFRESH_PORTFOLIO, self._auto_refresh_portfolio)
        self.set_interval(REFRESH_ALERTS,    self._check_alerts)

    # ── Auto-refresh callbacks ────────────────────────────────────────────────

    def _auto_refresh_markets(self) -> None:
        tab = self.query_one(TabbedContent)
        if tab.active == "markets":
            self.query_one(MarketsTab).load_markets()

    def _auto_refresh_whales(self) -> None:
        tab = self.query_one(TabbedContent)
        if tab.active == "whales":
            self.query_one(WhalesTab).load_whales()

    def _auto_refresh_arb(self) -> None:
        tab = self.query_one(TabbedContent)
        if tab.active == "arbitrage":
            self.query_one(ArbitrageTab).scan()

    def _auto_refresh_risk(self) -> None:
        tab = self.query_one(TabbedContent)
        if tab.active == "risk":
            self.query_one(RiskTab).load_risk()

    def _auto_refresh_portfolio(self) -> None:
        tab = self.query_one(TabbedContent)
        if tab.active == "portfolio":
            addr = self.query_one("#wallet-addr", Input).value.strip()
            if addr:
                self.query_one(PortfolioTab)._load_wallet(addr)

    async def _check_alerts(self) -> None:
        await self.query_one(AlertsTab).check_alerts()

    # ── Key-binding actions ───────────────────────────────────────────────────

    def action_refresh(self) -> None:
        active = self.query_one(TabbedContent).active
        dispatch = {
            "markets":   lambda: self.query_one(MarketsTab).load_markets(),
            "whales":    lambda: self.query_one(WhalesTab).load_whales(),
            "arbitrage": lambda: self.query_one(ArbitrageTab).scan(),
            "risk":      lambda: self.query_one(RiskTab).load_risk(),
            "bookmarks": lambda: self.query_one(BookmarksTab)._load(),
            "alerts":    lambda: self.query_one(AlertsTab)._load(),
            "journal":   lambda: self.query_one(JournalTab)._load(),
        }
        fn = dispatch.get(active)
        if fn:
            fn()

    def action_bookmark(self) -> None:
        if self.query_one(TabbedContent).active == "markets":
            self.query_one(MarketsTab).action_bookmark()

    def action_alert(self) -> None:
        if self.query_one(TabbedContent).active == "markets":
            self.query_one(MarketsTab).action_alert()

    def action_journal(self) -> None:
        active = self.query_one(TabbedContent).active
        if active == "markets":
            self.query_one(MarketsTab).action_journal()
        elif active == "journal":
            self.query_one(JournalTab).on_button_pressed(
                Button.Pressed(self.query_one("#jnl-add", Button))
            )

    def action_orderbook(self) -> None:
        if self.query_one(TabbedContent).active == "markets":
            self.query_one(MarketsTab).action_orderbook()

    def action_delete_selected(self) -> None:
        active = self.query_one(TabbedContent).active
        if active == "bookmarks":
            self.query_one(BookmarksTab).on_button_pressed(
                Button.Pressed(self.query_one("#bm-delete", Button))
            )
        elif active == "alerts":
            self.query_one(AlertsTab).on_button_pressed(
                Button.Pressed(self.query_one("#al-delete", Button))
            )
        elif active == "journal":
            self.query_one(JournalTab).on_button_pressed(
                Button.Pressed(self.query_one("#jnl-delete", Button))
            )

    async def on_unmount(self) -> None:
        await poly_client.close()
        await kalshi_client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S UTC")


def _selected_key(table: DataTable) -> str | None:
    """Return the row key of the currently-highlighted DataTable row, or None."""
    if table.cursor_row < 0 or table.row_count == 0:
        return None
    try:
        rows = table.ordered_rows
        if table.cursor_row < len(rows):
            key = rows[table.cursor_row].key
            return str(key.value) if key.value is not None else None
    except Exception:
        pass
    # Fallback via coordinate
    try:
        cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        return str(cell_key.row_key.value)
    except Exception:
        return None
