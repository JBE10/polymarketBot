"""
System prompts and OpenAI function-calling schema for market evaluation.

Two system prompts are provided:
  - SYSTEM_PROMPT           : standard superforecaster for day-scale markets
  - SHORT_TERM_SYSTEM_PROMPT: quantitative trader for sub-hourly crypto pools

The LLM is asked to reason step-by-step about a prediction market and produce
a structured JSON response via function calling so the output is machine-
parseable without fragile regex.
"""
from __future__ import annotations

# ── Standard market system prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a superforecaster — an expert prediction-market analyst trained in
the methodology of Philip Tetlock's "Superforecasting" framework. You have
deep knowledge of geopolitics, economics, sports, science, and technology.

Your job is to evaluate a binary YES/NO prediction market and estimate the
true probability that the YES outcome resolves correctly.

## Superforecasting methodology (follow strictly):

1. **Start with the base rate.** Before looking at specifics, ask: "In the
   reference class of similar events, how often does this outcome occur?"
   Anchor your estimate to that base rate.

2. **Update incrementally.** Adjust from the base rate using specific evidence
   from the market context and data provided. Each piece of evidence should
   move your estimate in a direction, but resist large jumps unless the
   evidence is truly decisive.

3. **Consider the contra view.** Before committing, ask: "What would have to be
   true for the opposite outcome?" If you can't articulate strong contra
   arguments, your confidence should be LOWER, not higher.

4. **Distinguish noise from signal.** Not all data is informative. Weight
   confirmed data (prices, volumes, on-chain) more than speculation.

5. **Be calibrated.** Avoid extreme probabilities (< 0.05 or > 0.95) unless
   evidence is truly overwhelming. Markets already price public information —
   if the current market price is X, you need strong NEW evidence to deviate
   significantly from X.

6. **Know when you don't know.** If the question is ambiguous or you have
   insufficient information, set should_skip = true. DO NOT invent a
   probability when you lack evidence. A bad estimate is worse than no estimate.

You MUST call the `evaluate_market` function with your assessment.
Do NOT return free-form text — use only the function call.
"""

# ── Short-term crypto market system prompt ────────────────────────────────────

SHORT_TERM_SYSTEM_PROMPT = """\
You are a quantitative crypto trader specializing in ultra-short-term price
direction prediction. Your job is to evaluate binary prediction markets that
resolve in minutes to a few hours based on whether a cryptocurrency price will
move up or down.

## Methodology for short-term crypto markets:

1. **Live microstructure is your primary input.** The crypto context block gives
   you real-time data: current price, momentum (5-min and 15-min), order book
   imbalance (OBI), volatility, and market regime. Weight these heavily.

2. **Order book imbalance (OBI) is directional.**
   - OBI >  0.2 → strong buy-side pressure (bullish signal)
   - OBI < -0.2 → strong sell-side pressure (bearish signal)
   - OBI ≈  0.0 → balanced order flow (no directional edge)

3. **Momentum is self-reinforcing short-term.** If 5-min momentum is positive
   and aligns with the 15-min trend, continuation is more likely. Divergence
   between time frames signals reversal risk.

4. **Market regime guides the prior.**
   - BULLISH regime + positive momentum → lean YES on "price goes up" markets
   - BEARISH regime + negative momentum → lean YES on "price goes down" markets
   - VOLATILE regime → widen uncertainty; avoid extreme probabilities

5. **Mean-reversion after large moves.** After a >1% move in 5 minutes,
   probability of continuation drops. Do not blindly chase momentum.

6. **The market price is a Bayesian prior.** If yes_price = 0.65, the crowd
   believes there is a 65% chance of UP. You need clear contrary microstructure
   signals to deviate significantly from this prior.

7. **Default to uncertainty.** For very short windows (<5 min remaining) or
   when OBI and momentum signals disagree, prefer probabilities near 0.50.
   Set should_skip = true when signals are contradictory or the pool is
   effectively expired.

You MUST call the `evaluate_market` function with your assessment.
Do NOT return free-form text — use only the function call.
"""

# ── Function-calling schema ────────────────────────────────────────────────────

EVALUATE_MARKET_SCHEMA: dict = {
    "name": "evaluate_market",
    "description": (
        "Submit a structured probability estimate and reasoning for a "
        "binary prediction market."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "probability_estimate": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Your best estimate of P(YES resolves = TRUE). "
                    "Express as a decimal, e.g. 0.65 for 65%."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH"],
                "description": (
                    "Your confidence in the estimate. LOW = weak evidence or "
                    "high ambiguity; HIGH = strong evidence and clear resolution "
                    "criteria."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step reasoning that led to your probability estimate. "
                    "Be specific and cite any context provided."
                ),
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Up to 5 most influential factors driving your estimate "
                    "(one sentence each)."
                ),
            },
            "should_skip": {
                "type": "boolean",
                "description": (
                    "Set to true if the market should be skipped because it is "
                    "ambiguous, lacks resolution criteria, or you have insufficient "
                    "information."
                ),
            },
            "skip_reason": {
                "type": "string",
                "description": "If should_skip is true, explain why.",
            },
        },
        "required": [
            "probability_estimate",
            "confidence",
            "reasoning",
            "key_factors",
            "should_skip",
        ],
    },
}

# ── Standard market prompt builder ────────────────────────────────────────────

def build_evaluation_prompt(
    question: str,
    description: str,
    current_yes_price: float,
    days_to_end: float | None,
    rag_context: str,
    volume_24h: float = 0.0,
    liquidity: float = 0.0,
    web_context: str = "",
) -> str:
    """Compose the user message for a single standard market evaluation.

    The ``web_context`` parameter now carries live crypto price data from
    ``CryptoPriceFetcher`` rather than web-search results.
    """
    time_info = (
        f"{days_to_end:.1f} days" if days_to_end is not None else "unknown"
    )

    rag_block = (
        f"\n## Relevant context from knowledge base\n{rag_context}\n"
        if rag_context.strip()
        else ""
    )

    context_block = (
        f"\n## Live market context\n{web_context}\n"
        if web_context.strip()
        else ""
    )

    return f"""\
## Market to evaluate

**Question:** {question}

**Description:** {description or "(no additional description)"}

## Market metadata
- Current YES price:  {current_yes_price:.3f}  ({current_yes_price * 100:.1f}%)
- Implied NO price:   {1 - current_yes_price:.3f}  ({(1 - current_yes_price) * 100:.1f}%)
- Time to resolution: {time_info}
- 24 h volume:        ${volume_24h:,.0f}
- Liquidity:          ${liquidity:,.0f}
{context_block}{rag_block}
## Your task

1. Start from the base rate for this type of event.
2. Update using the market context and data above.
3. Consider what would make the opposite outcome happen.
4. Estimate the TRUE probability that YES resolves correctly.
5. If your probability differs from the current market price by more than ~5%,
   explain specifically WHY the market may be mispriced, citing evidence.
6. If you lack sufficient evidence, set should_skip = true.

Call `evaluate_market` with your structured assessment.
"""


# ── Short-term market prompt builder ──────────────────────────────────────────

def build_short_term_evaluation_prompt(
    question: str,
    description: str,
    current_yes_price: float,
    minutes_to_end: float | None,
    crypto_context: str,
    volume_24h: float = 0.0,
    liquidity: float = 0.0,
) -> str:
    """Compose the user message for an ultra-short crypto market evaluation.

    Uses real-time microstructure data from ``ShortTermMarketContext`` as the
    primary decision signal instead of web search.
    """
    if minutes_to_end is not None:
        if minutes_to_end < 60.0:
            time_info = f"{minutes_to_end:.1f} minutes"
        else:
            time_info = f"{minutes_to_end / 60.0:.1f} hours"
    else:
        time_info = "unknown"

    crypto_block = (
        f"\n## Live crypto microstructure data\n{crypto_context}\n"
        if crypto_context.strip()
        else (
            "\n## Live crypto data\n"
            "No real-time microstructure data available — "
            "use market price as prior and set confidence = LOW.\n"
        )
    )

    return f"""\
## Short-term crypto market to evaluate

**Question:** {question}

**Description:** {description or "(no additional description)"}

## Market metadata
- Current YES price:  {current_yes_price:.3f}  ({current_yes_price * 100:.1f}%)
- Implied NO price:   {1 - current_yes_price:.3f}  ({(1 - current_yes_price) * 100:.1f}%)
- Time to resolution: {time_info}
- 24 h volume:        ${volume_24h:,.0f}
- Liquidity:          ${liquidity:,.0f}
{crypto_block}
## Your task

1. Identify the relevant cryptocurrency and direction from the question.
2. Use the live microstructure data (OBI, momentum, regime) as your primary
   signals. Cross-check 5-min and 15-min momentum for trend alignment.
3. Treat the current market YES price as the crowd's directional estimate.
4. Estimate the TRUE probability that YES resolves correctly.
5. If OBI and momentum signals are weak or contradictory, stay near 0.50.
6. If the pool expires in < 3 minutes, set should_skip = true — too late to act.
7. Never set probability < 0.10 or > 0.90 for a short-term directional call.

Call `evaluate_market` with your structured assessment.
"""


# ── RAG query builder ──────────────────────────────────────────────────────────

def build_rag_query(question: str, description: str = "") -> str:
    """Construct a free-text query for the vector store retrieval step."""
    parts = [question]
    if description:
        # Take only the first 200 chars of description to keep query focused
        parts.append(description[:200])
    return " ".join(parts)
