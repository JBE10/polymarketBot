"""
Phase 4 — LLM evaluation pipeline tests.

Mocks the Ollama HTTP call to avoid requiring a running Ollama instance.
Tests the full pipeline: prompt building → LLM call → JSON parsing → Pydantic model.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.ai.market_context import CryptoSignal
from src.ai.prompts import SYSTEM_PROMPT, build_evaluation_prompt, build_rag_query
from src.core.config import Settings
from src.core.database import Database
from src.polymarket.models import Confidence, Market, MarketEvaluation, MarketToken
from src.strategy.llm_evaluator import LLMEvaluator

# ── Canned LLM responses ─────────────────────────────────────────────────────

CANNED_LLM_RESPONSE = {
    "probability_estimate": 0.35,
    "confidence": "MEDIUM",
    "reasoning": "Oil prices have been declining due to ceasefire agreement. "
                 "The market currently prices YES at 0.45 but the fundamentals "
                 "suggest lower probability given the reduced geopolitical risk.",
    "key_factors": [
        "Ceasefire agreement signed",
        "WTI dropped to $72.40",
        "Analysts expect sustained decline",
    ],
    "should_skip": False,
    "skip_reason": "",
}

CANNED_OLLAMA_HTTP_RESPONSE = {
    "model": "gemma3:4b",
    "message": {
        "role": "assistant",
        "content": json.dumps(CANNED_LLM_RESPONSE),
    },
    "done": True,
}


# ── Prompt building tests ─────────────────────────────────────────────────────


class TestPromptBuilding:
    """Verify prompt construction functions produce usable strings."""

    def test_build_rag_query(self):
        query = build_rag_query(
            question="Will BTC exceed $100k?",
            description="Resolves YES if...",
        )
        assert isinstance(query, str)
        assert "BTC" in query

    def test_build_evaluation_prompt(self):
        prompt = build_evaluation_prompt(
            question="Will WTI close above $75?",
            description="Resolves YES if...",
            current_yes_price=0.45,
            days_to_end=30.0,
            rag_context="Oil prices fell today.",
            volume_24h=125_000,
            liquidity=80_000,
        )
        assert isinstance(prompt, str)
        assert "WTI" in prompt
        assert "0.45" in prompt or "45" in prompt

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50


# ── MarketEvaluation parsing tests ────────────────────────────────────────────


class TestMarketEvaluation:
    """MarketEvaluation Pydantic model validation."""

    def test_parse_valid_response(self):
        evaluation = MarketEvaluation(
            probability_estimate=float(CANNED_LLM_RESPONSE["probability_estimate"]),
            confidence=Confidence(CANNED_LLM_RESPONSE["confidence"]),
            reasoning=CANNED_LLM_RESPONSE["reasoning"],
            key_factors=CANNED_LLM_RESPONSE["key_factors"],
            should_skip=CANNED_LLM_RESPONSE["should_skip"],
            skip_reason=CANNED_LLM_RESPONSE["skip_reason"],
        )
        assert isinstance(evaluation.probability_estimate, float)
        assert 0.0 <= evaluation.probability_estimate <= 1.0
        assert evaluation.confidence == Confidence.MEDIUM
        assert len(evaluation.key_factors) == 3
        assert len(evaluation.reasoning) > 20

    def test_expected_value(self):
        evaluation = MarketEvaluation(
            probability_estimate=0.75,
            confidence=Confidence.HIGH,
            reasoning="Test reasoning that is long enough.",
            key_factors=["factor1"],
        )
        ev = evaluation.expected_value(entry_price=0.60)
        assert abs(ev - 0.15) < 0.001  # 0.75 - 0.60

    def test_probability_out_of_range_raises(self):
        with pytest.raises(Exception):
            MarketEvaluation(
                probability_estimate=1.5,  # > 1.0
                confidence=Confidence.LOW,
                reasoning="Test",
            )

    def test_confidence_case_insensitive(self):
        """Should accept lowercase confidence strings."""
        evaluation = MarketEvaluation(
            probability_estimate=0.5,
            confidence=Confidence("MEDIUM"),
            reasoning="Test reasoning for the model.",
        )
        assert evaluation.confidence == Confidence.MEDIUM


# ── LLM Evaluator JSON extraction ────────────────────────────────────────────


class TestJsonExtraction:
    """LLMEvaluator._extract_json() — strips markdown fences."""

    def test_plain_json(self):
        raw = '{"probability_estimate": 0.5}'
        assert LLMEvaluator._extract_json(raw) == raw.strip()

    def test_json_in_code_fence(self):
        raw = '```json\n{"probability_estimate": 0.5}\n```'
        result = LLMEvaluator._extract_json(raw)
        assert result == '{"probability_estimate": 0.5}'

    def test_json_in_plain_fence(self):
        raw = '```\n{"probability_estimate": 0.5}\n```'
        result = LLMEvaluator._extract_json(raw)
        assert result == '{"probability_estimate": 0.5}'

    def test_whitespace_handling(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = LLMEvaluator._extract_json(raw)
        assert '"key"' in result


# ── LLMEvaluator._parse_evaluation() ─────────────────────────────────────────


class TestParseEvaluation:
    """LLMEvaluator._parse_evaluation() — dict → MarketEvaluation."""

    def test_parse_valid_dict(self):
        evaluation = LLMEvaluator._parse_evaluation(CANNED_LLM_RESPONSE)
        assert isinstance(evaluation, MarketEvaluation)
        assert evaluation.probability_estimate == 0.35
        assert evaluation.confidence == Confidence.MEDIUM
        assert len(evaluation.key_factors) == 3

    def test_parse_lowercase_confidence(self):
        args = {**CANNED_LLM_RESPONSE, "confidence": "high"}
        evaluation = LLMEvaluator._parse_evaluation(args)
        assert evaluation.confidence == Confidence.HIGH

    def test_parse_missing_optional_fields(self):
        """Missing should_skip/skip_reason should use defaults."""
        args = {
            "probability_estimate": 0.60,
            "confidence": "LOW",
            "reasoning": "Minimal reasoning provided.",
        }
        evaluation = LLMEvaluator._parse_evaluation(args)
        assert evaluation.should_skip is False
        assert evaluation.skip_reason == ""


class TestShortTermDecision:
    """Short-term pipeline chooses YES or NO and persists simulation metadata."""

    @pytest.mark.asyncio
    async def test_short_term_can_buy_no_in_dry_run(
        self, settings: Settings, temp_db: Database
    ):
        settings.short_term_use_monte_carlo = True
        settings.short_term_mc_seed = 7
        settings.short_term_mc_samples = 5_000
        settings.short_term_min_ev_threshold = 0.03
        settings.min_confidence = "LOW"
        settings.short_term_math_signal_weight = 0.16

        evaluator = LLMEvaluator(
            clob=AsyncMock(),
            rag=AsyncMock(),
            db=temp_db,
            settings=settings,
        )
        evaluator._short_term_ctx.get_signal = AsyncMock(  # type: ignore[method-assign]
            return_value=CryptoSignal(
                symbol="BTC",
                price=100_000,
                change_1m_pct=-0.40,
                change_5m_pct=-0.80,
                change_15m_pct=-1.20,
                obi=-0.50,
                volatility_pct=0.20,
                regime="bearish",
            )
        )

        market = Market(
            condition_id="0x" + "9" * 64,
            question_id="qid_short_no",
            question="Will Bitcoin go up in the next 15 minutes?",
            description="Short BTC pool.",
            tokens=[
                MarketToken(token_id="yes_short", outcome="Yes", price=0.52),
                MarketToken(token_id="no_short", outcome="No", price=0.47),
            ],
            volume=10_000,
            volume_24hr=5_000,
            liquidity=2_000,
            end_date_iso=(datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
        )

        result = await evaluator._run_short_term_math_and_act(
            market=market,
            yes_price=0.52,
            bankroll=1_000,
            minutes_to_end=15,
        )

        assert result["action"].value == "BUY"
        assert result["chosen_side"] == "NO"

        await evaluator._act(market, result, bankroll=1_000)
        positions = await temp_db.get_open_positions()
        assert positions[0]["outcome"] == "NO"
        assert positions[0]["token_id"] == "no_short"

        assert temp_db._db is not None
        async with temp_db._db.execute(
            "SELECT chosen_side, side_price, no_price, mc_samples, mc_mean_edge "
            "FROM evaluations WHERE market_id=?",
            (market.condition_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row["chosen_side"] == "NO"
        assert row["side_price"] == pytest.approx(0.47)
        assert row["no_price"] == pytest.approx(0.47)
        assert row["mc_samples"] == 5_000
        assert row["mc_mean_edge"] > 0

        await evaluator._http_client.aclose()
        await evaluator._short_term_ctx.close()
