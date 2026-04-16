# Polymarket Intelligence

An advanced, terminal-based analytics and intelligence dashboard for [Polymarket](https://polymarket.com) prediction markets — built with Python and [Textual](https://textual.textualize.io/).

> **View-only tool** — no private keys are ever requested or stored. All analytics are read-only.

---

## Features

| Tab | What it does |
|-----|--------------|
| **Markets** | Live feed of active markets sorted by 24 h volume. Search, bookmark, and jump to order books. |
| **Whales** | Detect large trades (≥ $5 k) across top markets in real time. |
| **Arbitrage** | Cross-platform price comparison between Polymarket and Kalshi with spread and profit estimates. |
| **Order Book** | ASCII depth chart for any market — bids, asks, mid-price, and spread visualized. |
| **Risk** | Composite risk scores (liquidity, activity, time-to-resolution, certainty) for every market. |
| **Portfolio** | Enter any wallet address (0x…) to view open positions and estimated P&L. |
| **Bookmarks** | Saved markets with personal notes. |
| **Alerts** | Price threshold alerts checked automatically every 30 s. |
| **Journal** | Personal trade journal — record observations, rationale, outcomes, and P&L. |

---

## Installation

```bash
# Requires Python 3.10+
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Or without activating the venv:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

---

## Bot Quick Commands

Run the bot in dry-run mode (using current `.env`):

```bash
cd /Users/juanbautistaespino/Documents/polymarketBot && source .venv/bin/activate && python main.py
```

Run A/B paper mode with dump+hedge 15m enabled (keeps current bot + adds DH15M loop):

```bash
cd /Users/juanbautistaespino/Documents/polymarketBot && source .venv/bin/activate && DH15M_ENABLED=true DRY_RUN=true python -m src.main
```

Open a live terminal dashboard (orders, win rate, P&L, balance estimate, incidents):

```bash
cd /Users/juanbautistaespino/Documents/polymarketBot && source .venv/bin/activate && python scripts/terminal_dashboard.py --repo . --health-port 8080 --refresh 2
```

Compute daily combined performance (LLM + MM), including trades, win rate, and net P&L:

```bash
cd /Users/juanbautistaespino/Documents/polymarketBot && { rg --no-filename "Position CLOSED .*realized_pnl=" logs/bot.log* | perl -ne 'if(/^(\d{4}-\d{2}-\d{2}).*realized_pnl=([+-]?\d+\.\d+)/){print "$1 $2\n"}'; rg --no-filename "\[MM\] SELL FILLED" logs/bot.log* | perl -ne 'if(/^(\d{4}-\d{2}-\d{2}).*total=([+-]?\d+\.\d+)/){print "$1 $2\n"}'; rg --no-filename "\[MM\] (STOP-LOSS|TIME-STOP)" logs/bot.log* | perl -ne 'if(/^(\d{4}-\d{2}-\d{2}).*pnl=([+-]?\d+\.\d+)/){print "$1 $2\n"}'; } | awk 'NF==2{d=$1;p=$2+0;n[d]++;s[d]+=p;if(p>0)w[d]++;else if(p<0)l[d]++} END{for (d in n){wr=(n[d]?100*w[d]/n[d]:0);printf("%s trades=%d wr=%.1f%% net=%+.4f\n",d,n[d],wr,s[d])}}' | sort
```

Run the bot in background (recommended for long sessions):

```bash
nohup /Users/juanbautistaespino/Documents/polymarketBot/.venv/bin/python /Users/juanbautistaespino/Documents/polymarketBot/main.py >> /Users/juanbautistaespino/Documents/polymarketBot/logs/bot.log 2>&1 & echo $! > /Users/juanbautistaespino/Documents/polymarketBot/.bot.pid
```

Check if it is active:

```bash
pgrep -af "polymarketBot/main.py"
```

Watch logs live:

```bash
tail -f /Users/juanbautistaespino/Documents/polymarketBot/logs/bot.log
```

Stop the bot cleanly:

```bash
kill "$(cat /Users/juanbautistaespino/Documents/polymarketBot/.bot.pid)"
```

About windows:
- The terminal where you run `python main.py` is the bot process in foreground.
- A terminal with `tail -f logs/bot.log` is only a live log viewer.
- The README editor window is documentation only.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Move between tabs |
| `r` | Refresh current tab |
| `b` | Bookmark selected market |
| `a` | Add price alert for selected market |
| `o` | Open order book for selected market |
| `j` | Add journal entry |
| `d` | Delete selected row (Bookmarks / Alerts / Journal) |
| `q` | Quit |

---

## Data Sources

- **Polymarket Gamma API** — market listings, prices, liquidity
- **Polymarket CLOB API** — order books, recent trades
- **Polymarket Data API** — wallet positions
- **Kalshi Trading API** — market listings for arbitrage comparison (public endpoints, no auth required)

---

## Local Storage

All personal data is stored in a local SQLite database at `data/polymarket_intelligence.db`.

| Table | Contents |
|-------|----------|
| `bookmarks` | Saved markets + notes |
| `alerts` | Price threshold alerts |
| `journal` | Trade journal entries |
| `wallets` | Saved wallet addresses (labels only) |
| `api_cache` | Short-lived API response cache |

---

## Architecture

```
polymarketBot/
├── main.py            Entry point
├── config.py          Constants & settings
├── database/
│   └── db.py          SQLite manager
├── api/
│   ├── polymarket.py  Async Polymarket client
│   └── kalshi.py      Async Kalshi client
├── analytics/
│   ├── whale.py       Whale trade detection
│   ├── arbitrage.py   Cross-platform arb finder
│   └── risk.py        Market risk scoring
└── ui/
    ├── charts.py      ASCII chart utilities
    └── app.py         Textual TUI application
```
