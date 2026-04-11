# Project Evaluation and Change Plan (API-First)

## Goal
Evaluate the full project end-to-end and execute changes safely, with measurable gates for promotion from planning to implementation.

## Baseline Status (Today)
- Strategy specification is defined in `docs/strategy_spec.md`.
- Core test execution was blocked by a legacy import path (`src.core.database`).
- A compatibility bridge was added in `src/core/database.py`.

## Phase 0 - Baseline and Inventory (Day 1)
Objective: get a reliable baseline and map architecture drift.

Tasks:
- Run smoke checks:
  - import checks for core modules
  - unit tests for model/risk/position sizing
  - one integration test for orchestration path
- Collect failures into an issue list with:
  - severity (`critical`, `high`, `medium`)
  - owner module (`clients`, `model`, `execution`, etc.)
  - reproduction command
- Confirm runtime config loading from `strategy_config.yaml` by environment profile (`demo`, `real`).

Exit criteria:
- Test baseline report generated.
- Top 10 blockers prioritized.

## Phase 1 - Config and Contract Hardening (Days 1-2)
Objective: lock config schema and module contracts before behavioral changes.

Tasks:
- Extend typed config models to cover all runtime sections currently present in YAML:
  - `data_sources`, `ranking`, `execution`, `safety`, `logging`, `profiles`
- Add strict validation and default guards:
  - thresholds in valid ranges
  - risk limits monotonic (`real` tighter than `demo`)
  - required per-asset keys for enabled universe
- Add contract tests:
  - config load by profile
  - invalid config rejection
  - deterministic merge behavior

Exit criteria:
- Config schema complete and validated.
- Contract tests green.

## Phase 2 - Strategy Core Consistency (Days 2-4)
Objective: align implementation with strategy spec mathematics and filters.

Tasks:
- Verify edge equation consistency in `model/` and `orchestrator/`.
- Ensure no-trade window and spread/depth gates are enforced centrally.
- Normalize units (probability vs percent) to avoid mixed-scale bugs.
- Verify ranker formula and top-N asset selection behavior.

Exit criteria:
- Mathematical parity checklist complete.
- Unit tests for formulas and gates green.

## Phase 3 - Execution and Resilience (Days 4-6)
Objective: harden order lifecycle and operational safety.

Tasks:
- Enforce idempotency keys and deduplication policy.
- Validate cancel/requote flow (`max_requotes`, TIF).
- Add bounded retry with jitter for API transient failures.
- Wire kill-switch triggers to safe halt path.

Exit criteria:
- Failure-mode integration tests green.
- No duplicate-order scenarios in chaos tests.

## Phase 4 - Observability and Replay (Days 6-7)
Objective: make every decision auditable and replayable.

Tasks:
- Persist decision snapshots:
  - features
  - p_model / p_market / costs / edge
  - filters pass/fail
  - order outcomes
- Add daily reporting dimensions:
  - asset, regime, hour-of-day, side
- Build replay utility from snapshots for deterministic decision re-run.

Exit criteria:
- Replay of a sample session reproduces decision outputs.
- Daily report artifact generated successfully.

## Phase 5 - Paper Validation Gate (Week 2+)
Objective: validate readiness for real capital using strict quantitative gates.

Tasks:
- Paper run with production-equivalent infrastructure.
- Track promotion metrics from spec:
  - >= 150 paper trades
  - EV/trade > 0
  - Profit Factor > 1.15
  - Max DD within limits
  - no severe operational incidents

Exit criteria:
- Written promotion report with pass/fail per metric.

## Operating Rules During Refactor
- No changes to core formulas without matching spec update.
- Every parameter change must include rationale and evidence.
- No silent fallback for risk or execution failures.

## Immediate Next Actions
1. Stabilize test entrypoint and run full baseline.
2. Harden `src/core/yaml_config.py` schema coverage.
3. Create config validation tests under `tests/`.
4. Open implementation batch for highest severity blockers.
