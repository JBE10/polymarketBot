"""
System prompts and OpenAI function-calling schema for market evaluation.

The LLM is asked to reason step-by-step about a prediction market and produce
a structured JSON response via function calling so the output is machine-
parseable without fragile regex.
"""
from __future__ import annotations

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert prediction-market analyst with deep knowledge of:
  • Geopolitics, economics, sports, science, and technology trends
  • Bayesian probability estimation and base-rate reasoning
  • How prediction markets price risk and information

Your job is to evaluate a binary YES/NO prediction market and estimate the
true probability that the YES outcome resolves correctly.

Reasoning guidelines:
  1. Anchor to base rates and reference classes first.
  2. Update based on specific evidence from the provided context.
  3. Apply inside-view adjustments (expert consensus, recent data).
  4. Account for market-specific risks: resolution ambiguity, time horizon.
  5. Be calibrated — avoid extreme probabilities (< 0.05 or > 0.95) unless the
     evidence is overwhelming.
  6. If the question is ambiguous, poorly defined, or you lack sufficient
     information to form a credible view, set should_skip = true.

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

# ── User prompt builder ────────────────────────────────────────────────────────

def build_evaluation_prompt(
    question: str,
    description: str,
    current_yes_price: float,
    days_to_end: float | None,
    rag_context: str,
    volume_24h: float = 0.0,
    liquidity: float = 0.0,
) -> str:
    """Compose the user message for a single market evaluation."""

    time_info = (
        f"{days_to_end:.1f} days" if days_to_end is not None else "unknown"
    )

    context_block = (
        f"\n## Relevant context from knowledge base\n{rag_context}\n"
        if rag_context.strip()
        else "\n## Relevant context\nNo relevant documents found.\n"
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
{context_block}
## Your task

Estimate the TRUE probability that YES resolves correctly.
If your probability differs from the current market price by more than ~3 %, 
explain why the market may be mispriced.

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
