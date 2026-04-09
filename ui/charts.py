"""
ASCII chart utilities for the terminal dashboard.
All functions return Rich markup strings — safe to pass to Textual's Static / RichLog.
"""
from __future__ import annotations


# ── Order-book depth chart ────────────────────────────────────────────────────

def order_book_chart(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    rows: int = 8,
    bar_width: int = 28,
) -> str:
    """
    Render a two-sided ASCII order-book depth chart.

    Parameters
    ----------
    bids  : list of (price, size) sorted best→worst (descending price)
    asks  : list of (price, size) sorted best→worst (ascending price)
    rows  : how many levels to show per side
    bar_width : max character width for the bar
    """
    if not bids and not asks:
        return "[dim italic]  No order-book data available.[/dim italic]"

    bids = bids[:rows]
    asks = asks[:rows]

    all_sizes = [s for _, s in bids] + [s for _, s in asks]
    max_size  = max(all_sizes) if all_sizes else 1.0

    def bar(size: float, color: str) -> str:
        n = max(1, int(size / max_size * bar_width))
        return f"[{color}]{'█' * n}{'░' * (bar_width - n)}[/{color}]"

    lines: list[str] = []
    lines.append(
        "[bold cyan]┌──────────────────────────────────────────────────────┐[/bold cyan]"
    )
    lines.append(
        "[bold cyan]│           ORDER BOOK DEPTH (USD)                     │[/bold cyan]"
    )
    lines.append(
        "[bold cyan]├────────────┬────────────────────────────────┬─────────┤[/bold cyan]"
    )

    # Asks (reversed so lowest ask is at mid-line)
    lines.append("[bold red]│    ASKS    │                                │         │[/bold red]")
    for price, size in reversed(asks):
        lines.append(
            f"[red]│   {price:6.4f}  │[/red]{bar(size,'red')}[dim]│{size:>8,.0f} │[/dim]"
        )

    # Mid / spread
    if bids and asks:
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid      = (best_bid + best_ask) / 2
        spread   = best_ask - best_bid
        pct      = spread / mid * 100 if mid else 0
        lines.append(
            f"[bold yellow]│  MID {mid:6.4f} │ spread {spread:.4f}  ({pct:.2f} %)         │[/bold yellow]"
        )
    else:
        lines.append(
            "[bold yellow]│ ──────────────────────────────────────────────────── │[/bold yellow]"
        )

    # Bids
    lines.append("[bold green]│    BIDS    │                                │         │[/bold green]")
    for price, size in bids:
        lines.append(
            f"[green]│   {price:6.4f}  │[/green]{bar(size,'green')}[dim]│{size:>8,.0f} │[/dim]"
        )

    lines.append(
        "[bold cyan]└──────────────────────────────────────────────────────┘[/bold cyan]"
    )
    return "\n".join(lines)


# ── Sparkline ─────────────────────────────────────────────────────────────────

_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(prices: list[float], width: int = 36) -> str:
    """Return a single-line Unicode sparkline coloured by trend."""
    if len(prices) < 2:
        return "─" * width

    lo, hi = min(prices), max(prices)
    span   = hi - lo or 1.0

    step    = max(1, len(prices) // width)
    sampled = prices[::step][:width]
    line    = "".join(
        _SPARK_BLOCKS[min(int((p - lo) / span * (len(_SPARK_BLOCKS) - 1)), 8)]
        for p in sampled
    )

    color = "green" if prices[-1] >= prices[0] else "red"
    return f"[{color}]{line}[/{color}]"


# ── Sentiment bar ─────────────────────────────────────────────────────────────

def sentiment_bar(buy_pct: float, width: int = 22) -> str:
    """Horizontal buy/sell bar with percentage labels."""
    buy_n  = max(0, min(width, int(buy_pct / 100 * width)))
    sell_n = width - buy_n
    bar    = f"[green]{'█' * buy_n}[/green][red]{'█' * sell_n}[/red]"
    return f"{bar}  [green]{buy_pct:.0f}% B[/green] [dim]/[/dim] [red]{100 - buy_pct:.0f}% S[/red]"


# ── Risk gauge ────────────────────────────────────────────────────────────────

def risk_gauge(score: float, width: int = 18) -> str:
    """Filled progress bar coloured by risk level."""
    filled = max(0, min(width, int(score / 100 * width)))
    empty  = width - filled

    if score < 30:
        color, label = "green",  "LOW "
    elif score < 65:
        color, label = "yellow", "MED "
    else:
        color, label = "red",    "HIGH"

    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
    return f"{bar} [{color}]{score:4.0f}[/{color}] [dim]{label}[/dim]"


# ── Currency / price formatters ───────────────────────────────────────────────

def fmt_usd(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.2f}"


def fmt_price(price: float | None) -> str:
    if price is None:
        return "  N/A  "
    return f"{price:.3f} ({price * 100:.1f}%)"


def fmt_pnl(pnl: float | None) -> str:
    if pnl is None:
        return "[dim]—[/dim]"
    color = "green" if pnl >= 0 else "red"
    sign  = "+" if pnl >= 0 else ""
    return f"[{color}]{sign}{pnl:,.2f}[/{color}]"


def price_color(price: float | None, prev: float | None = None) -> str:
    """Return rich color tag for a price: green if up, red if down."""
    if price is None:
        return "dim"
    if prev is None:
        return "white"
    return "green" if price >= prev else "red"
