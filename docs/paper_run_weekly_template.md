# Paper Trading Weekly Template (Demo -> Real)

## Objective
Run a strict 2-4 week paper phase with measurable promotion criteria:
- >= 150 trades
- EV/trade > 0
- Profit Factor > 1.15
- Max Drawdown <= 2.2%
- Severe incidents = 0

## Daily Routine (UTC)

### 1. Start Session
- Start bot in dry-run with market-making enabled:
```bash
MM_ENABLED=true DRY_RUN=true CYCLE_INTERVAL_SECONDS=10 MM_CYCLE_SECONDS=2 /Users/juanbautistaespino/Documents/polymarketBot/.venv/bin/python -m src.main
```

### 2. Mid-Session Checks (every 2-4h)
- Check promotion metrics snapshot:
```bash
/Users/juanbautistaespino/Documents/polymarketBot/.venv/bin/python scripts/promotion_gate_report.py
```
- Check segment report outputs (asset/regime/hour/side) when needed.

### 3. End Session
- Stop bot cleanly.
- Export promotion gate report JSON.
- Export daily decision report CSVs.

## Weekly KPI Targets

### Week 1 target
- Trades: 35-45
- EV/trade: >= -0.20 (exploratory tolerance)
- Profit Factor: >= 0.95
- Max DD: <= 3.0%
- Severe incidents: 0

### Week 2 target
- Cumulative trades: 80-100
- EV/trade: >= -0.05
- Profit Factor: >= 1.05
- Max DD: <= 2.7%
- Severe incidents: 0

### Week 3 target
- Cumulative trades: 120-140
- EV/trade: >= 0.00
- Profit Factor: >= 1.12
- Max DD: <= 2.4%
- Severe incidents: 0

### Week 4 target (promotion window)
- Cumulative trades: >= 150
- EV/trade: > 0.00
- Profit Factor: > 1.15
- Max DD: <= 2.2%
- Severe incidents: 0

## Daily Log Template

Date (UTC):
- Runtime hours:
- Trades today:
- Cumulative trades:
- EV/trade:
- Profit Factor:
- Max DD %:
- Severe incidents:
- Top failing check today:
- Action for next day:

## Decision Rules

### Continue as-is
- At least 4/5 metrics on trajectory and severe incidents remain 0.

### Tighten risk for next day
- EV/trade negative for 2 consecutive days.
- Profit Factor below 1.0 for 2 consecutive days.
- DD worsening trend for 2 consecutive days.

### Halt and investigate
- Any severe incident.
- DD breach above 2.2% in paper mode.
- Data/feed anomalies causing repeated kill-switch activations.

## Promotion Checklist
- Final report from `scripts/promotion_gate_report.py`
- Daily CSV reports present for last 7 days
- Incident log reviewed and signed off
- Parameter changes documented with rationale
