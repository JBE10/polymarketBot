"""
LLM-powered strategy orchestrator.

Cycle (called once per bot iteration):
    1. Fetch active markets from Polymarket CLOB.
    2. Filter candidates (liquidity, time horizon, not already evaluated/open).
    3. For each candidate:
        a. Query RAG for relevant context.
        b. Call LLM (Ollama local o Gemini API) → structured MarketEvaluation.
        c. Run Kelly/EV mathematics.
        d. If EV > threshold and confidence >= min_confidence:
             – In dry-run mode: log only.
             – In live mode: place order, persist to DB.
    4. Refresh open position prices and check for exit conditions.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx

from src.ai.prompts import (
    EVALUATE_MARKET_SCHEMA,
    SYSTEM_PROMPT,
    build_evaluation_prompt,
    build_rag_query,
)
from src.ai.rag_engine import RagEngine
from src.ai.web_search import WebSearcher
from src.core.config import Settings
from src.core.database import Database
from src.polymarket.clob_client import AsyncClobClient
from src.polymarket.models import (
    Action,
    Confidence,
    Market,
    MarketEvaluation,
    OrderRequest,
    Side,
)
from src.strategy.kelly import compute_kelly

log = logging.getLogger(__name__)


# ── Candidate filters ──────────────────────────────────────────────────────────

_MIN_LIQUIDITY_USD  = 5_000.0    # skip illiquid markets
_MIN_DAYS_REMAINING = 2.0        # skip markets resolving in < 2 days
_MAX_DAYS_REMAINING = 90.0       # skip very long-horizon markets (high uncertainty)
_MIN_VOLUME_24H_USD = 500.0      # skip low-activity markets

# ── JSON schema string for Ollama prompt ───────────────────────────────────────

_JSON_SCHEMA_INSTRUCTIONS = """\
You MUST respond with ONLY a valid JSON object (no markdown fences, no extra text).
The JSON must have exactly these fields:

{
  "probability_estimate": <float 0.0–1.0>,
  "confidence": "<LOW | MEDIUM | HIGH>",
  "reasoning": "<step-by-step reasoning string>",
  "key_factors": ["<factor 1>", "<factor 2>", ...],
  "should_skip": <true | false>,
  "skip_reason": "<string, empty if should_skip is false>"
}
"""


class LLMEvaluator:
    """
    Main strategy engine.  Stateless between calls — all persistence goes
    through the Database instance.
    """

    def __init__(
        self,
        clob:     AsyncClobClient,
        rag:      RagEngine,
        db:       Database,
        settings: Settings,
    ) -> None:
        self._clob = clob
        self._rag  = rag
        self._db   = db
        self._cfg  = settings

        # Markets approved for market-making (populated after each cycle)
        self._approved_markets: dict[str, Market] = {}

        # ── Web search engine ──────────────────────────────────────────────────
        self._searcher = WebSearcher(
            api_key=settings.tavily_api_key,
            max_results=settings.web_search_max_results,
            enabled=settings.web_search_enabled,
        )
        if self._searcher.is_available:
            log.info("Web search enabled (Tavily, max_results=%d)", settings.web_search_max_results)
        else:
            log.info("Web search disabled (no TAVILY_API_KEY or web_search_enabled=false)")

        # ── Configure LLM backend ─────────────────────────────────────────────
        self._provider = settings.llm_provider  # "ollama" | "gemini" | "claude"

        if self._provider == "gemini":
            from google import genai
            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
            self._gemini_model  = settings.gemini_model
        elif self._provider == "claude":
            self._anthropic_key   = settings.anthropic_api_key
            self._anthropic_model = settings.anthropic_model
            self._http_client     = httpx.AsyncClient(timeout=120.0)
            log.info("Using Claude API — model: %s", self._anthropic_model)
        else:
            self._ollama_base_url = settings.ollama_base_url.rstrip("/")
            self._ollama_model    = settings.ollama_model
            self._http_client     = httpx.AsyncClient(timeout=120.0)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_approved_markets(self) -> dict[str, Market]:
        """Markets approved by the LLM for trading (updated each cycle)."""
        return dict(self._approved_markets)

    async def run_cycle(self, bankroll: float) -> dict[str, int]:
        """
        Execute one full evaluation cycle.

        Returns a summary dict: {evaluated, acted, skipped, errors}.
        """
        summary = {"evaluated": 0, "acted": 0, "skipped": 0, "errors": 0}

        # ── 1. Fetch markets ──────────────────────────────────────────────────
        markets = await self._clob.get_markets(limit=100)
        log.info("Fetched %d markets from CLOB.", len(markets))
        markets_by_id = {m.condition_id: m for m in markets}

        # ── 2. Refresh prices + check exits BEFORE evaluating new markets ─────
        await self._refresh_positions()
        exits = await self._check_exits(markets_by_id)
        if exits:
            log.info("Closed %d position(s) this cycle.", exits)

        candidates = await self._filter_candidates(markets)
        log.info("%d candidates pass pre-filters.", len(candidates))

        # ── 3. Evaluate each candidate ────────────────────────────────────────
        approved_this_cycle: dict[str, Market] = {}

        for market in candidates:
            try:
                result = await self._evaluate_market(market, bankroll)
                summary["evaluated"] += 1

                if result["action"] == Action.BUY:
                    await self._act(market, result, bankroll)
                    summary["acted"] += 1
                    approved_this_cycle[market.condition_id] = market
                else:
                    summary["skipped"] += 1

            except Exception as exc:
                log.exception("Error evaluating market %s: %s", market.condition_id, exc)
                summary["errors"] += 1

        # Merge into approved set (keep previous approvals that are still valid)
        for mid in list(self._approved_markets):
            if mid not in markets_by_id:
                del self._approved_markets[mid]
        self._approved_markets.update(approved_this_cycle)

        log.info(
            "Cycle complete — evaluated=%d acted=%d skipped=%d errors=%d exits=%d",
            summary["evaluated"], summary["acted"],
            summary["skipped"], summary["errors"], exits,
        )
        return summary

    # ── Pre-filtering ─────────────────────────────────────────────────────────

    async def _filter_candidates(self, markets: list[Market]) -> list[Market]:
        """Apply fast, cheap filters before the expensive LLM call."""
        open_positions = {p["market_id"] for p in await self._db.get_open_positions()}

        candidates = []
        for m in markets:
            # Skip if already in a position
            if m.condition_id in open_positions:
                continue

            # Skip if recently evaluated (de-duplicate across cycles)
            if await self._db.was_recently_evaluated(m.condition_id, within_hours=4):
                continue

            # Liquidity and activity gate
            if m.liquidity < _MIN_LIQUIDITY_USD:
                continue
            if m.volume_24hr < _MIN_VOLUME_24H_USD:
                continue

            # Time horizon gate
            days = m.days_to_end
            if days is not None:
                if days < _MIN_DAYS_REMAINING or days > _MAX_DAYS_REMAINING:
                    continue

            # Must have a YES token with a valid price
            if m.yes_price is None or not (0.03 < m.yes_price < 0.97):
                continue

            candidates.append(m)

        return candidates

    # ── LLM evaluation ────────────────────────────────────────────────────────

    async def _evaluate_market(
        self, market: Market, bankroll: float
    ) -> dict:
        """
        Full pipeline for a single market:
        RAG retrieval → LLM call → Kelly sizing → DB log.
        """
        yes_price = market.yes_price or 0.5
        days      = market.days_to_end

        # ── Web search (real-time context) ─────────────────────────────────────
        web_context = ""
        if self._searcher.is_available:
            search_query = f"{market.question} {market.description[:100]}".strip()
            web_context = await self._searcher.search(search_query)
            if web_context:
                log.info("Web search: %d chars for '%s'", len(web_context), market.question[:40])
                # Persist web results into ChromaDB for future RAG retrieval
                try:
                    await self._rag.add_document(
                        text=web_context,
                        source="web_search",
                        title=f"Web: {market.question[:80]}",
                    )
                except Exception:
                    pass

        # ── RAG retrieval ─────────────────────────────────────────────────────
        rag_query   = build_rag_query(market.question, market.description)
        rag_context = await self._rag.retrieve(rag_query, top_k=5)

        # ── Build prompt ──────────────────────────────────────────────────────
        user_msg = build_evaluation_prompt(
            question=market.question,
            description=market.description,
            current_yes_price=yes_price,
            days_to_end=days,
            rag_context=rag_context,
            volume_24h=market.volume_24hr,
            liquidity=market.liquidity,
            web_context=web_context,
        )

        # ── LLM call (dispatch by provider) ───────────────────────────────────
        if self._provider == "claude":
            evaluation = await self._call_claude(user_msg)
        elif self._provider == "ollama":
            evaluation = await self._call_ollama(user_msg)
        else:
            evaluation = await self._call_gemini(user_msg)

        if evaluation is None:
            return {"action": Action.SKIP, "skip_reason": "LLM call failed"}

        # ── Kelly / EV sizing ─────────────────────────────────────────────────
        kelly = compute_kelly(
            prob=evaluation.probability_estimate,
            entry_price=yes_price,
            bankroll=bankroll,
            kelly_fraction=self._cfg.kelly_fraction,
            max_position_usd=self._cfg.max_position_usd,
        )

        # ── Decision ─────────────────────────────────────────────────────────
        action = self._decide(evaluation, kelly)

        # ── Persist evaluation ─────────────────────────────────────────────────
        await self._db.insert_evaluation(
            market_id=market.condition_id,
            question=market.question,
            market_price=yes_price,
            estimated_prob=evaluation.probability_estimate,
            expected_value=kelly.ev_per_dollar,
            kelly_fraction=kelly.frac_kelly,
            position_size_usd=kelly.position_usd,
            confidence=evaluation.confidence.value,
            reasoning=evaluation.reasoning,
            key_factors=json.dumps(evaluation.key_factors),
            action=action.value,
            skip_reason=evaluation.skip_reason,
        )

        # ── Calibration tracking ──────────────────────────────────────────────
        await self._db.upsert_calibration(
            market_id=market.condition_id,
            question=market.question,
            predicted_prob=evaluation.probability_estimate,
            market_price=yes_price,
            llm_provider=self._provider,
        )

        log.info(
            "Evaluated: %-55s  price=%.3f  prob=%.3f  EV=%+.3f  action=%s",
            market.question[:55],
            yes_price,
            evaluation.probability_estimate,
            kelly.ev_per_dollar,
            action.value,
        )

        return {
            "action":     action,
            "evaluation": evaluation,
            "kelly":      kelly,
            "market":     market,
            "yes_price":  yes_price,
        }

    # ── Ollama (local) ────────────────────────────────────────────────────────

    async def _call_ollama(self, user_message: str) -> Optional[MarketEvaluation]:
        """Call Ollama local API and parse structured JSON response."""
        content = ""
        try:
            # Append explicit JSON instructions to the user message — more reliable
            # than `format: "json"` for small models like gemma3:4b which return {}
            closing = (
                "\n\nRespond ONLY with a JSON object. No prose, no markdown fences. "
                "Required keys: probability_estimate (float 0-1), "
                "confidence (LOW|MEDIUM|HIGH), reasoning (string), "
                "key_factors (array of strings), should_skip (boolean), "
                "skip_reason (string).\n\nJSON:"
            )

            payload = {
                "model": self._ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message + closing},
                ],
                "stream": False,
                # Note: "format": "json" is intentionally omitted — gemma3:4b
                # returns empty JSON {} when it is set, but works correctly
                # when the instruction appears in the user message instead.
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                },
            }

            url = f"{self._ollama_base_url}/api/chat"
            log.info("Calling Ollama (%s) …", self._ollama_model)

            response = await self._http_client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data.get("message", {}).get("content", "")

            if not content.strip():
                log.warning("Ollama returned empty content — skipping market.")
                return None

            json_str = self._extract_json(content)
            args = json.loads(json_str)

            if not args or "probability_estimate" not in args:
                log.warning("Ollama JSON missing required fields: %s", content[:300])
                return None

            return self._parse_evaluation(args)

        except httpx.HTTPStatusError as exc:
            log.error("Ollama HTTP error %d: %s", exc.response.status_code, exc.response.text[:200])
            return None
        except json.JSONDecodeError as exc:
            log.error("Failed to parse Ollama JSON: %s\nRaw: %s", exc, content[:500])
            return None
        except Exception as exc:
            log.error("Ollama call failed: %s", exc)
            return None

    # ── Gemini (Google API) ───────────────────────────────────────────────────

    async def _call_gemini(self, user_message: str) -> Optional[MarketEvaluation]:
        """Call Gemini with function-calling and parse the structured response."""
        try:
            from google.genai import types as genai_types

            tool = genai_types.Tool(
                function_declarations=[EVALUATE_MARKET_SCHEMA]
            )
            config = genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[tool],
                tool_config=genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["evaluate_market"],
                    )
                ),
                temperature=0.2,
            )

            response = await self._gemini_client.aio.models.generate_content(
                model=self._gemini_model,
                contents=user_message,
                config=config,
            )

            # Extract function call from response
            args = None
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name == "evaluate_market":
                        args = {k: v for k, v in fc.args.items()}
                        break
                if args is not None:
                    break

            if args is None:
                log.warning("Gemini no devolvió un function call.")
                return None

            return self._parse_evaluation(args)

        except Exception as exc:
            log.error("Gemini call failed: %s", exc)
            return None

    # ── Claude (Anthropic API) ────────────────────────────────────────────

    async def _call_claude(self, user_message: str) -> Optional[MarketEvaluation]:
        """Call Claude via Anthropic Messages API and parse structured JSON."""
        content = ""
        try:
            closing = (
                "\n\nRespond ONLY with a valid JSON object. No prose, no markdown fences.\n"
                "Required keys: probability_estimate (float 0-1), "
                "confidence (LOW|MEDIUM|HIGH), reasoning (string), "
                "key_factors (array of strings), should_skip (boolean), "
                "skip_reason (string).\n\nJSON:"
            )

            payload = {
                "model": self._anthropic_model,
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_message + closing},
                ],
                "temperature": 0.2,
            }

            log.info("Calling Claude (%s) …", self._anthropic_model)

            response = await self._http_client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self._anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()

            data = response.json()
            blocks = data.get("content", [])
            content = "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )

            if not content.strip():
                log.warning("Claude returned empty content — skipping market.")
                return None

            json_str = self._extract_json(content)
            args = json.loads(json_str)

            if not args or "probability_estimate" not in args:
                log.warning("Claude JSON missing required fields: %s", content[:300])
                return None

            return self._parse_evaluation(args)

        except httpx.HTTPStatusError as exc:
            log.error("Claude HTTP error %d: %s", exc.response.status_code, exc.response.text[:200])
            return None
        except json.JSONDecodeError as exc:
            log.error("Failed to parse Claude JSON: %s\nRaw: %s", exc, content[:500])
            return None
        except Exception as exc:
            log.error("Claude call failed: %s", exc)
            return None

    # ── Shared parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences if present and return raw JSON string."""
        text = text.strip()
        # Remove ```json ... ``` wrappers
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def _parse_evaluation(args: dict) -> MarketEvaluation:
        """Parse a dict of LLM output into a MarketEvaluation model."""
        return MarketEvaluation(
            probability_estimate=float(args["probability_estimate"]),
            confidence=Confidence(str(args.get("confidence", "LOW")).upper()),
            reasoning=str(args.get("reasoning", "")),
            key_factors=list(args.get("key_factors", [])),
            should_skip=bool(args.get("should_skip", False)),
            skip_reason=str(args.get("skip_reason", "")),
        )

    # ── Decision gate ─────────────────────────────────────────────────────────

    def _decide(self, ev: MarketEvaluation, kelly) -> Action:
        """Apply all thresholds to determine whether to BUY or SKIP."""
        log.debug(
            "_decide: EV=%.3f pos=$%.2f conf=%s should_skip=%s min_conf=%s",
            kelly.ev_per_dollar, kelly.position_usd,
            ev.confidence.value, ev.should_skip, self._cfg.min_confidence,
        )

        # Hard math gates first — if the numbers don't work, always skip
        if not kelly.is_positive_ev:
            return Action.SKIP

        if kelly.ev_per_dollar < self._cfg.min_ev_threshold:
            return Action.SKIP

        if kelly.position_usd < 1.0:   # never bother for < $1
            log.info("SKIP: position too small ($%.2f)", kelly.position_usd)
            return Action.SKIP

        # Confidence gate
        meets = self._cfg.meets_confidence(ev.confidence.value)
        log.info("Confidence check: level=%s min=%s meets=%s", ev.confidence.value, self._cfg.min_confidence, meets)
        if not meets:
            return Action.SKIP

        # LLM's qualitative skip recommendation only applies when EV is marginal
        # (< 2× the minimum threshold). Strong EV overrides a soft skip signal.
        if ev.should_skip and kelly.ev_per_dollar < self._cfg.min_ev_threshold * 2:
            return Action.SKIP

        return Action.BUY

    # ── Order execution ───────────────────────────────────────────────────────

    async def _act(self, market: Market, result: dict, bankroll: float) -> None:
        """Place an order (or simulate in dry-run) and update the database."""
        kelly    = result["kelly"]
        yes_price = result["yes_price"]
        yes_token = market.yes_token

        if yes_token is None:
            log.warning("No YES token for %s — skipping.", market.condition_id)
            return

        shares   = kelly.shares
        cost_usd = kelly.position_usd

        if self._cfg.dry_run:
            log.info(
                "[DRY-RUN] Would BUY %.2f YES shares of '%s' at %.3f ($%.2f)",
                shares, market.question[:50], yes_price, cost_usd,
            )
            await self._db.insert_order(
                market_id=market.condition_id,
                token_id=yes_token.token_id,
                question=market.question,
                side="BUY",
                price=yes_price,
                size=shares,
                status="DRY_RUN",
            )
            # Record position so exit logic can track it next cycle
            await self._db.upsert_position(
                market_id=market.condition_id,
                token_id=yes_token.token_id,
                question=market.question,
                outcome="YES",
                avg_entry_price=yes_price,
                shares=shares,
                cost_usd=cost_usd,
                current_price=yes_price,
            )
            return

        # ── Live order ─────────────────────────────────────────────────────────
        order_req = OrderRequest(
            token_id=yes_token.token_id,
            price=round(yes_price, 2),   # round to tick
            size=round(shares, 2),
            side=Side.BUY,
        )

        order_db_id = await self._db.insert_order(
            market_id=market.condition_id,
            token_id=yes_token.token_id,
            question=market.question,
            side="BUY",
            price=yes_price,
            size=shares,
        )

        resp = await self._clob.place_order(order_req)

        await self._db.update_order_status(
            order_id=order_db_id,
            status=resp.status.value,
            transaction_hash=resp.transaction_hash,
            error_message=resp.error_message,
        )

        if resp.success:
            await self._db.upsert_position(
                market_id=market.condition_id,
                token_id=yes_token.token_id,
                question=market.question,
                outcome="YES",
                avg_entry_price=yes_price,
                shares=resp.filled_size or shares,
                cost_usd=cost_usd,
                current_price=yes_price,
            )
            log.info(
                "Order FILLED: %.2f YES shares of '%s' @ %.3f (txh=%s)",
                resp.filled_size, market.question[:45],
                yes_price, (resp.transaction_hash or "N/A")[:12],
            )
        else:
            log.warning(
                "Order FAILED for '%s': %s",
                market.question[:45], resp.error_message,
            )

    # ── Position maintenance ──────────────────────────────────────────────────

    async def _refresh_positions(self) -> None:
        """Update current prices for all open positions."""
        positions = await self._db.get_open_positions()
        if not positions:
            return

        for pos in positions:
            try:
                book = await self._clob.get_order_book(pos["token_id"])
                if book and book.mid_price is not None:
                    await self._db.update_position_price(
                        pos["market_id"], book.mid_price
                    )
            except Exception as exc:
                log.debug(
                    "Could not refresh price for %s: %s", pos["market_id"], exc
                )

    async def _check_exits(self, markets_by_id: dict) -> int:
        """
        Check every open position against exit conditions and close if triggered.

        Exit conditions (checked in priority order):
          1. Take-profit  : current_price >= entry * (1 + take_profit_pct)
          2. Stop-loss    : current_price <= entry * (1 - stop_loss_pct)
          3. Time exit    : days_to_resolution <= exit_days_before_end

        Returns the number of positions closed this cycle.
        """
        positions = await self._db.get_open_positions()
        if not positions:
            return 0

        tp  = self._cfg.take_profit_pct
        sl  = self._cfg.stop_loss_pct
        tde = self._cfg.exit_days_before_end
        closed = 0

        for pos in positions:
            market_id    = pos["market_id"]
            entry        = pos["avg_entry_price"]
            current      = pos.get("current_price") or entry
            shares       = pos["shares"]
            cost_usd     = pos["cost_usd"]
            question     = pos["question"][:55]

            # ── 1. Take-profit ────────────────────────────────────────────────
            if current >= entry * (1 + tp):
                reason    = f"take-profit +{tp*100:.0f}%"
                pnl       = shares * current - cost_usd
                pnl_pct   = pnl / cost_usd * 100 if cost_usd else 0
                await self._sell_position(pos, current, reason)
                log.info(
                    "✅ EXIT [%s] '%s'\n"
                    "   entry=%.3f  exit=%.3f  pnl=%+.2f (%+.1f%%)",
                    reason, question, entry, current, pnl, pnl_pct,
                )
                closed += 1
                continue

            # ── 2. Stop-loss ──────────────────────────────────────────────────
            if current <= entry * (1 - sl):
                reason    = f"stop-loss -{sl*100:.0f}%"
                pnl       = shares * current - cost_usd
                pnl_pct   = pnl / cost_usd * 100 if cost_usd else 0
                await self._sell_position(pos, current, reason)
                log.info(
                    "🛑 EXIT [%s] '%s'\n"
                    "   entry=%.3f  exit=%.3f  pnl=%+.2f (%+.1f%%)",
                    reason, question, entry, current, pnl, pnl_pct,
                )
                closed += 1
                continue

            # ── 3. Time-based exit ────────────────────────────────────────────
            market = markets_by_id.get(market_id)
            if market and market.days_to_end is not None:
                if market.days_to_end <= tde:
                    reason  = f"time-exit ({market.days_to_end:.1f}d left)"
                    pnl     = shares * current - cost_usd
                    pnl_pct = pnl / cost_usd * 100 if cost_usd else 0
                    await self._sell_position(pos, current, reason)
                    log.info(
                        "⏰ EXIT [%s] '%s'\n"
                        "   entry=%.3f  exit=%.3f  pnl=%+.2f (%+.1f%%)",
                        reason, question, entry, current, pnl, pnl_pct,
                    )
                    closed += 1

        return closed

    async def _sell_position(self, pos: dict, exit_price: float, reason: str) -> None:
        """Place a SELL order (or simulate) and close the position record."""
        market_id = pos["market_id"]
        token_id  = pos["token_id"]
        shares    = pos["shares"]
        question  = pos["question"]

        if self._cfg.dry_run:
            proceeds  = shares * exit_price
            cost_usd  = pos["cost_usd"]
            pnl       = proceeds - cost_usd
            log.info(
                "[DRY-RUN] Would SELL %.2f shares of '%s' at %.3f "
                "→ proceeds=$%.2f  pnl=%+.2f  reason=%s",
                shares, question[:45], exit_price, proceeds, pnl, reason,
            )
        else:
            order_req = OrderRequest(
                token_id=token_id,
                price=round(exit_price, 2),
                size=round(shares, 2),
                side=Side.SELL,
            )
            await self._db.insert_order(
                market_id=market_id,
                token_id=token_id,
                question=question,
                side="SELL",
                price=exit_price,
                size=shares,
            )
            resp = await self._clob.place_order(order_req)
            if not resp.success:
                log.warning("SELL order failed for %s: %s", market_id, resp.error_message)
                return

        await self._db.close_position(
            market_id=market_id,
            exit_price=exit_price,
            exit_reason=reason,
        )
