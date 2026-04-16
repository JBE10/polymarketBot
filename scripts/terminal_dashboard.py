from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Nothing-inspired terminal palette: monochrome, high contrast, minimal accents.
PALETTE = {
    "fg": "bright_white",
    "muted": "grey70",
    "dim": "bright_black",
    "ok": "white",
    "warn": "grey70",
    "bad": "grey50",
    "border": "grey50",
}

RE_REALIZED = re.compile(r"realized_pnl=([+-]?\d+\.\d+)")
RE_MM_TOTAL = re.compile(r"\[MM\] SELL FILLED: .*total=([+-]?\d+\.\d+)")
RE_MM_STOP = re.compile(r"\[MM\] (?:STOP-LOSS|TIME-STOP): .*pnl=([+-]?\d+\.\d+)")
RE_BANKROLL = re.compile(r"^\s*bankroll_usd\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
RE_ENV_FLOAT = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)


def _tail(value: str, n: int = 48) -> str:
    if not value:
        return ""
    if len(value) <= n:
        return value
    return f"...{value[-n:]}"


def _safe_float(v: Any) -> float:
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


def _read_bankroll(strategy_cfg_path: Path) -> float:
    if not strategy_cfg_path.exists():
        return 0.0
    text = strategy_cfg_path.read_text(encoding="utf-8", errors="ignore")
    m = RE_BANKROLL.search(text)
    if not m:
        return 0.0
    return float(m.group(1))


def _read_env_float(env_path: Path, key: str, default: float) -> float:
    if not env_path.exists():
        return default
    text = env_path.read_text(encoding="utf-8", errors="ignore")
    values = {m.group(1): float(m.group(2)) for m in RE_ENV_FLOAT.finditer(text)}
    return float(values.get(key, default))


def _probe_health(url: str, timeout_sec: float = 1.5) -> tuple[bool, str]:
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            body = resp.read(64).decode("utf-8", errors="ignore").strip()
            code = getattr(resp, "status", 200)
            ok = 200 <= int(code) < 300
            return ok, body or str(code)
    except URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def _process_status(pattern: str) -> tuple[bool, int, str]:
    proc = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return False, 0, ""
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return True, len(lines), lines[0]


def _collect_log_metrics(log_dir: Path) -> dict[str, float | int]:
    files = sorted(log_dir.glob("bot.log*"))
    llm_vals: list[float] = []
    mm_vals: list[float] = []

    for path in files:
        try:
            for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = RE_REALIZED.search(raw)
                if m:
                    llm_vals.append(float(m.group(1)))
                    continue

                m = RE_MM_TOTAL.search(raw)
                if m:
                    mm_vals.append(float(m.group(1)))
                    continue

                m = RE_MM_STOP.search(raw)
                if m:
                    mm_vals.append(float(m.group(1)))
                    continue
        except Exception:
            continue

    combined = llm_vals + mm_vals

    def agg(values: list[float]) -> tuple[int, int, int, float, float]:
        n = len(values)
        w = sum(1 for v in values if v > 0)
        l = sum(1 for v in values if v < 0)
        s = sum(values)
        wr = (100.0 * w / n) if n else 0.0
        return n, w, l, s, wr

    llm_n, llm_w, llm_l, llm_net, llm_wr = agg(llm_vals)
    mm_n, mm_w, mm_l, mm_net, mm_wr = agg(mm_vals)
    all_n, all_w, all_l, all_net, all_wr = agg(combined)

    return {
        "llm_n": llm_n,
        "llm_w": llm_w,
        "llm_l": llm_l,
        "llm_net": llm_net,
        "llm_wr": llm_wr,
        "mm_n": mm_n,
        "mm_w": mm_w,
        "mm_l": mm_l,
        "mm_net": mm_net,
        "mm_wr": mm_wr,
        "all_n": all_n,
        "all_w": all_w,
        "all_l": all_l,
        "all_net": all_net,
        "all_wr": all_wr,
    }


def _query_db(db_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "open_positions": 0,
        "open_cost": 0.0,
        "open_value": 0.0,
        "open_unrealized": 0.0,
        "orders": [],
        "mm_active": 0,
        "mm_closed": 0,
        "mm_rejections_24h": [],
        "incidents_24h": [],
        "dh_events_24h": 0,
        "dh_cycles_24h": 0,
        "dh_hedged_24h": 0,
        "dh_stop_hedged_24h": 0,
        "dh_avg_sum_price_24h": 0.0,
        "dh_settled_24h": 0,
        "dh_avg_gross_ev_24h": 0.0,
    }
    if not db_path.exists():
        return out

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                COALESCE(SUM(cost_usd), 0) AS cost,
                COALESCE(SUM(current_value_usd), 0) AS val,
                COALESCE(SUM(unrealized_pnl), 0) AS upnl
            FROM positions
            WHERE status='OPEN'
            """
        )
        row = cur.fetchone()
        if row:
            out["open_positions"] = int(row["n"] or 0)
            out["open_cost"] = _safe_float(row["cost"])
            out["open_value"] = _safe_float(row["val"])
            out["open_unrealized"] = _safe_float(row["upnl"])

        cur = conn.execute(
            """
            SELECT id, side, status, question, price, size, created_at
            FROM orders
            ORDER BY created_at DESC
            LIMIT 12
            """
        )
        out["orders"] = [dict(r) for r in cur.fetchall()]

        cur = conn.execute(
            "SELECT COUNT(*) AS n FROM mm_rounds WHERE status IN ('BUY_POSTED','BOUGHT','SELL_POSTED')"
        )
        out["mm_active"] = int((cur.fetchone() or {"n": 0})["n"])

        cur = conn.execute("SELECT COUNT(*) AS n FROM mm_rounds WHERE status='CLOSED'")
        out["mm_closed"] = int((cur.fetchone() or {"n": 0})["n"])

        cur = conn.execute(
            """
            SELECT reason_code, COUNT(*) AS cnt
            FROM mm_rejections
            WHERE created_at >= datetime('now', '-24 hours')
            GROUP BY reason_code
            ORDER BY cnt DESC, reason_code ASC
            LIMIT 8
            """
        )
        out["mm_rejections_24h"] = [dict(r) for r in cur.fetchall()]

        cur = conn.execute(
            """
            SELECT severity, incident_type, message, ts_utc
            FROM operational_incidents
            WHERE ts_utc >= datetime('now', '-24 hours')
            ORDER BY ts_utc DESC
            LIMIT 8
            """
        )
        out["incidents_24h"] = [dict(r) for r in cur.fetchall()]

        cur = conn.execute(
            """
            SELECT
                COUNT(*) AS events,
                COUNT(DISTINCT cycle_id) AS cycles,
                SUM(CASE WHEN status='HEDGED' THEN 1 ELSE 0 END) AS hedged,
                SUM(CASE WHEN status='STOP_HEDGED' THEN 1 ELSE 0 END) AS stop_hedged,
                AVG(CASE WHEN sum_price IS NOT NULL THEN sum_price END) AS avg_sum
            FROM dump_hedge_events
            WHERE created_at >= datetime('now', '-24 hours')
            """
        )
        row = cur.fetchone()
        if row:
            out["dh_events_24h"] = int(row["events"] or 0)
            out["dh_cycles_24h"] = int(row["cycles"] or 0)
            out["dh_hedged_24h"] = int(row["hedged"] or 0)
            out["dh_stop_hedged_24h"] = int(row["stop_hedged"] or 0)
            out["dh_avg_sum_price_24h"] = _safe_float(row["avg_sum"])

        cur = conn.execute(
            """
            SELECT
                COUNT(*) AS settled,
                AVG(1.0 - sum_price) AS avg_gross_ev
            FROM dump_hedge_events
            WHERE created_at >= datetime('now', '-24 hours')
              AND status IN ('HEDGED', 'STOP_HEDGED')
              AND sum_price IS NOT NULL
            """
        )
        row = cur.fetchone()
        if row:
            out["dh_settled_24h"] = int(row["settled"] or 0)
            out["dh_avg_gross_ev_24h"] = _safe_float(row["avg_gross_ev"])
    except sqlite3.Error:
        pass
    finally:
        conn.close()

    return out


def _build_header(health_ok: bool, health_msg: str, bot: tuple[bool, int, str], dash: tuple[bool, int, str]) -> Panel:
    bot_ok, bot_n, bot_line = bot
    dash_ok, dash_n, dash_line = dash

    t = Table.grid(padding=(0, 2))
    t.add_column(justify="left")
    t.add_column(justify="left")

    health_color = PALETTE["ok"] if health_ok else PALETTE["bad"]
    bot_color = PALETTE["ok"] if bot_ok else PALETTE["bad"]
    dash_color = PALETTE["ok"] if dash_ok else PALETTE["warn"]

    t.add_row("HEALTH", f"[{health_color}] {'UP' if health_ok else 'DOWN'} [/{health_color}]  [{PALETTE['muted']}]{health_msg}[/{PALETTE['muted']}]")
    t.add_row("BOT", f"[{bot_color}] {'RUNNING' if bot_ok else 'STOPPED'} [/{bot_color}]  [{PALETTE['muted']}]count={bot_n}  {_tail(bot_line)}[/{PALETTE['muted']}]")
    t.add_row("DASH", f"[{dash_color}] {'RUNNING' if dash_ok else 'STOPPED'} [/{dash_color}]  [{PALETTE['muted']}]count={dash_n}  {_tail(dash_line)}[/{PALETTE['muted']}]")

    return Panel(t, title="RUNTIME", border_style=PALETTE["border"], box=box.SQUARE)


def _build_kpi_panel(logm: dict[str, float | int], dbm: dict[str, Any], bankroll: float) -> Panel:
    all_net = _safe_float(logm.get("all_net"))
    est_balance = bankroll + all_net if bankroll > 0 else 0.0

    t = Table.grid(padding=(0, 2))
    t.add_column(style=f"bold {PALETTE['fg']}")
    t.add_column(justify="right")
    t.add_column(justify="right")

    t.add_row("Combined", f"trades={int(logm.get('all_n', 0))}", f"WR={_safe_float(logm.get('all_wr')):5.2f}%  net={all_net:+.2f}")
    t.add_row("LLM", f"trades={int(logm.get('llm_n', 0))}", f"WR={_safe_float(logm.get('llm_wr')):5.2f}%  net={_safe_float(logm.get('llm_net')):+.2f}")
    t.add_row("MM", f"trades={int(logm.get('mm_n', 0))}", f"WR={_safe_float(logm.get('mm_wr')):5.2f}%  net={_safe_float(logm.get('mm_net')):+.2f}")
    t.add_row("Open Positions", str(int(dbm.get("open_positions", 0))), f"uPnL={_safe_float(dbm.get('open_unrealized')):+.2f}")
    t.add_row("Open Exposure", f"cost={_safe_float(dbm.get('open_cost')):.2f}", f"value={_safe_float(dbm.get('open_value')):.2f}")

    if bankroll > 0:
        t.add_row("Bankroll", f"{bankroll:.2f}", f"est.balance={est_balance:.2f}")

    t.add_row("MM Rounds", f"active={int(dbm.get('mm_active', 0))}", f"closed={int(dbm.get('mm_closed', 0))}")
    t.add_row("DH15M", f"events24h={int(dbm.get('dh_events_24h', 0))}", f"cycles24h={int(dbm.get('dh_cycles_24h', 0))}")

    return Panel(t, title="METRICS", border_style=PALETTE["border"], box=box.SQUARE)


def _build_orders_panel(orders: list[dict[str, Any]]) -> Panel:
    t = Table(box=box.SIMPLE)
    t.add_column("ts", no_wrap=True)
    t.add_column("side", no_wrap=True)
    t.add_column("status", no_wrap=True)
    t.add_column("price", justify="right")
    t.add_column("size", justify="right")
    t.add_column("question")

    for o in orders[:12]:
        side = str(o.get("side", ""))
        status = str(o.get("status", ""))
        ts = str(o.get("created_at", ""))[-8:]
        price = f"{_safe_float(o.get('price')):.3f}"
        size = f"{_safe_float(o.get('size')):.2f}"
        q = str(o.get("question", ""))[:64]
        side_color = PALETTE["fg"] if side.upper() == "BUY" else PALETTE["muted"]
        t.add_row(ts, f"[{side_color}]{side}[/{side_color}]", status, price, size, q)

    if not orders:
        t.add_row("-", "-", "-", "-", "-", "No orders yet")

    return Panel(t, title="ORDERS", border_style=PALETTE["border"], box=box.SQUARE)


def _build_rejections_panel(rows: list[dict[str, Any]]) -> Panel:
    t = Table(box=box.MINIMAL)
    t.add_column("reason")
    t.add_column("count", justify="right")
    if rows:
        for r in rows:
            t.add_row(str(r.get("reason_code", "?")), str(int(r.get("cnt", 0))))
    else:
        t.add_row("-", "0")
    return Panel(t, title="MM REJECTIONS 24H", border_style=PALETTE["border"], box=box.SQUARE)


def _build_incidents_panel(rows: list[dict[str, Any]]) -> Panel:
    t = Table(box=box.MINIMAL)
    t.add_column("ts", no_wrap=True)
    t.add_column("sev", no_wrap=True)
    t.add_column("type", no_wrap=True)
    t.add_column("message")

    for r in rows[:8]:
        sev = str(r.get("severity", "INFO"))
        sev_color = (
            PALETTE["bad"]
            if sev.upper() == "SEVERE"
            else (PALETTE["warn"] if sev.upper() == "WARN" else PALETTE["ok"])
        )
        t.add_row(
            str(r.get("ts_utc", ""))[-8:],
            f"[{sev_color}]{sev}[/{sev_color}]",
            str(r.get("incident_type", ""))[:20],
            str(r.get("message", ""))[:74],
        )

    if not rows:
        t.add_row("-", "-", "-", "No incidents in last 24h")

    return Panel(t, title="INCIDENTS 24H", border_style=PALETTE["border"], box=box.SQUARE)


def _build_dh15m_panel(dbm: dict[str, Any], *, cost_buffer_per_pair: float) -> Panel:
    cycles = int(dbm.get("dh_cycles_24h", 0))
    hedged = int(dbm.get("dh_hedged_24h", 0))
    stop_hedged = int(dbm.get("dh_stop_hedged_24h", 0))
    avg_sum = _safe_float(dbm.get("dh_avg_sum_price_24h", 0.0))
    settled = int(dbm.get("dh_settled_24h", 0))
    avg_gross_ev = _safe_float(dbm.get("dh_avg_gross_ev_24h", 0.0))
    avg_net_ev = avg_gross_ev - cost_buffer_per_pair
    est_net_total = settled * avg_net_ev

    hedge_rate = (100.0 * hedged / cycles) if cycles else 0.0
    stop_rate = (100.0 * stop_hedged / cycles) if cycles else 0.0

    t = Table.grid(padding=(0, 2))
    t.add_column(style=f"bold {PALETTE['fg']}")
    t.add_column(justify="right")
    t.add_column(justify="right")

    t.add_row("Cycles 24h", str(cycles), f"events={int(dbm.get('dh_events_24h', 0))}")
    t.add_row("Hedged", str(hedged), f"hedge_rate={hedge_rate:.1f}%")
    t.add_row("Stop-Hedged", str(stop_hedged), f"stop_rate={stop_rate:.1f}%")
    t.add_row("Avg Sum Price", f"{avg_sum:.4f}", "target<=sum_target")
    t.add_row("Avg Gross EV/trade", f"{avg_gross_ev:+.4f}", "1 - sum_price")
    t.add_row("Avg Net EV/trade", f"{avg_net_ev:+.4f}", f"minus cost={cost_buffer_per_pair:.4f}")
    t.add_row("Est Net EV 24h", f"{est_net_total:+.4f}", f"settled={settled}")

    return Panel(t, title="DH15M A/B 24H", border_style=PALETTE["border"], box=box.SQUARE)


def render_frame(repo_root: Path, health_port: int) -> Panel:
    db_path = repo_root / "data" / "bot_state.db"
    log_dir = repo_root / "logs"
    strategy_cfg = repo_root / "strategy_config.yaml"
    env_path = repo_root / ".env"

    bot = _process_status(r"src\.main|/main\.py")
    dash = _process_status(r"streamlit run src/dashboard\.py|streamlit")
    health_ok, health_msg = _probe_health(f"http://127.0.0.1:{health_port}/health")

    logm = _collect_log_metrics(log_dir)
    dbm = _query_db(db_path)
    bankroll = _read_bankroll(strategy_cfg)
    dh_cost = _read_env_float(env_path, "DH15M_COST_BUFFER_PER_PAIR", 0.01)

    group = Group(
        Panel("[grey70]N O T H I N G   O P S   V I E W[/grey70]", border_style=PALETTE["dim"], box=box.SQUARE),
        _build_header(health_ok, health_msg, bot, dash),
        _build_kpi_panel(logm, dbm, bankroll),
        _build_dh15m_panel(dbm, cost_buffer_per_pair=dh_cost),
        _build_orders_panel(dbm.get("orders", [])),
        _build_rejections_panel(dbm.get("mm_rejections_24h", [])),
        _build_incidents_panel(dbm.get("incidents_24h", [])),
    )
    return Panel(group, title="POLYMARKET TERMINAL", border_style=PALETTE["dim"], box=box.SQUARE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live terminal dashboard for bot ops metrics")
    parser.add_argument("--repo", default=".", help="Repository root path")
    parser.add_argument("--health-port", type=int, default=8080, help="Health endpoint port")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo).resolve()

    with Live(render_frame(repo_root, args.health_port), refresh_per_second=4, screen=True) as live:
        try:
            while True:
                time.sleep(max(args.refresh, 0.5))
                live.update(render_frame(repo_root, args.health_port))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
