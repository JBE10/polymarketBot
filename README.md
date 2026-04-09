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
