"""
Test: Compare LLM evaluations with and without web search context.

Fetches 5 live markets from Polymarket, evaluates each one TWICE
(once without web context, once with), and prints a comparison table.
The web-search version should produce probabilities closer to the
current market price or more confidently diverge with cited evidence.

Usage:
    .venv/bin/python tests/test_web_vs_blind.py

Requires:
    - Ollama running with gemma3:4b (or whichever LLM_PROVIDER is set)
    - TAVILY_API_KEY in .env (for the web-search pass)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from src.ai.prompts import SYSTEM_PROMPT, build_evaluation_prompt
from src.ai.web_search import WebSearcher
from src.core.config import get_settings

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("test_web_vs_blind")


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


async def fetch_markets(n: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 50, "active": "true", "closed": "false"},
        )
        resp.raise_for_status()
        raw = resp.json()

    candidates = []
    for m in raw:
        vol = float(m.get("volume24hr") or m.get("volume_24hr") or 0)
        liq = float(m.get("liquidity") or 0)
        price_str = m.get("outcomePrices") or m.get("outcome_prices") or "[]"
        try:
            prices = json.loads(price_str) if isinstance(price_str, str) else price_str
            yes_price = float(prices[0]) if prices else 0.0
        except Exception:
            yes_price = 0.0

        if liq >= 5000 and vol >= 500 and 0.10 < yes_price < 0.90:
            candidates.append({
                "question": m.get("question", ""),
                "description": m.get("description", "")[:200],
                "yes_price": yes_price,
            })

    return candidates[:n]


async def call_ollama(prompt: str, cfg) -> dict | None:
    closing = (
        "\n\nRespond ONLY with a valid JSON object. No prose, no markdown fences.\n"
        "Required keys: probability_estimate (float 0-1), "
        "confidence (LOW|MEDIUM|HIGH), reasoning (string), "
        "key_factors (array of strings), should_skip (boolean), "
        "skip_reason (string).\n\nJSON:"
    )
    payload = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + closing},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{cfg.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content.strip():
        return None
    try:
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return None


async def main() -> None:
    cfg = get_settings()
    searcher = WebSearcher(
        api_key=cfg.tavily_api_key,
        max_results=cfg.web_search_max_results,
        enabled=cfg.web_search_enabled,
    )

    print("=" * 72)
    print("  Test: Web Search vs. Blind LLM Evaluation")
    print("=" * 72)
    print(f"  LLM: {cfg.llm_provider} / {cfg.ollama_model}")
    print(f"  Web search: {'AVAILABLE' if searcher.is_available else 'DISABLED (no TAVILY_API_KEY)'}")
    print()

    if not searcher.is_available:
        print("  [SKIP] Set TAVILY_API_KEY in .env to run this test.")
        return

    markets = await fetch_markets(5)
    print(f"  Fetched {len(markets)} eligible markets.\n")

    results = []

    for i, mkt in enumerate(markets, 1):
        q = mkt["question"]
        print(f"  [{i}] {q[:65]}...")
        print(f"      Market price: {mkt['yes_price']:.3f}")

        # Pass 1: BLIND (no web context)
        prompt_blind = build_evaluation_prompt(
            question=q,
            description=mkt["description"],
            current_yes_price=mkt["yes_price"],
            days_to_end=None,
            rag_context="",
            web_context="",
        )
        print("      Evaluating BLIND ...", end="", flush=True)
        blind = await call_ollama(prompt_blind, cfg)
        blind_prob = blind.get("probability_estimate", -1) if blind else -1
        print(f" prob={blind_prob:.3f}" if blind_prob >= 0 else " FAILED")

        # Pass 2: WITH WEB SEARCH
        web_ctx = await searcher.search(f"{q} {mkt['description'][:100]}")
        prompt_web = build_evaluation_prompt(
            question=q,
            description=mkt["description"],
            current_yes_price=mkt["yes_price"],
            days_to_end=None,
            rag_context="",
            web_context=web_ctx,
        )
        print("      Evaluating WITH WEB ...", end="", flush=True)
        webbed = await call_ollama(prompt_web, cfg)
        web_prob = webbed.get("probability_estimate", -1) if webbed else -1
        print(f" prob={web_prob:.3f}" if web_prob >= 0 else " FAILED")

        delta_blind = abs(blind_prob - mkt["yes_price"]) if blind_prob >= 0 else None
        delta_web = abs(web_prob - mkt["yes_price"]) if web_prob >= 0 else None

        results.append({
            "question": q[:55],
            "market": mkt["yes_price"],
            "blind": blind_prob,
            "web": web_prob,
            "delta_blind": delta_blind,
            "delta_web": delta_web,
            "web_reasoning": (webbed.get("reasoning", "")[:120] + "...") if webbed else "",
        })
        print()

    # Summary table
    print("=" * 72)
    print(f"  {'Market':<55}  {'Price':>5}  {'Blind':>5}  {'Web':>5}  {'Closer?':>7}")
    print("-" * 72)
    web_wins = 0
    for r in results:
        if r["delta_blind"] is not None and r["delta_web"] is not None:
            closer = "WEB" if r["delta_web"] <= r["delta_blind"] else "BLIND"
            if closer == "WEB":
                web_wins += 1
        else:
            closer = "N/A"
        print(
            f"  {r['question']:<55}  {r['market']:>5.3f}  "
            f"{r['blind']:>5.3f}  {r['web']:>5.3f}  {closer:>7}"
        )
    print("-" * 72)
    total = len([r for r in results if r["delta_blind"] is not None and r["delta_web"] is not None])
    print(f"  Web search closer to market in {web_wins}/{total} cases")
    print("=" * 72)

    await searcher.close()


if __name__ == "__main__":
    asyncio.run(main())
