"""
Test: Compare Claude Sonnet vs gemma3:4b on the same markets.

Evaluates the same 5 markets with both models and compares how
close each model's predicted probability is to the current market
price (which already encodes public information).

Usage:
    .venv/bin/python tests/test_claude_vs_gemma.py

Requires:
    - Ollama running with gemma3:4b
    - ANTHROPIC_API_KEY in .env
    - (Optional) TAVILY_API_KEY for web context
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


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


_JSON_CLOSING = (
    "\n\nRespond ONLY with a valid JSON object. No prose, no markdown fences.\n"
    "Required keys: probability_estimate (float 0-1), "
    "confidence (LOW|MEDIUM|HIGH), reasoning (string), "
    "key_factors (array of strings), should_skip (boolean), "
    "skip_reason (string).\n\nJSON:"
)


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
    payload = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + _JSON_CLOSING},
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


async def call_claude(prompt: str, cfg) -> dict | None:
    if not cfg.anthropic_api_key:
        return None

    payload = {
        "model": cfg.anthropic_model,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": prompt + _JSON_CLOSING},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": cfg.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    blocks = data.get("content", [])
    content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
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

    print("=" * 78)
    print("  Test: Claude Sonnet vs gemma3:4b Calibration Comparison")
    print("=" * 78)
    print(f"  Ollama model : {cfg.ollama_model}")
    print(f"  Claude model : {cfg.anthropic_model}")
    print(f"  Claude API   : {'AVAILABLE' if cfg.anthropic_api_key else 'MISSING (set ANTHROPIC_API_KEY)'}")
    print(f"  Web search   : {'ON' if searcher.is_available else 'OFF'}")
    print()

    if not cfg.anthropic_api_key:
        print("  [SKIP] Set ANTHROPIC_API_KEY in .env to run this test.")
        print("  Tip: Get a key at https://console.anthropic.com/")
        return

    markets = await fetch_markets(5)
    print(f"  Fetched {len(markets)} eligible markets.\n")

    results = []

    for i, mkt in enumerate(markets, 1):
        q = mkt["question"]
        print(f"  [{i}] {q[:65]}...")
        print(f"      Market price: {mkt['yes_price']:.3f}")

        # Get web context if available
        web_ctx = ""
        if searcher.is_available:
            web_ctx = await searcher.search(f"{q} {mkt['description'][:100]}")

        prompt = build_evaluation_prompt(
            question=q,
            description=mkt["description"],
            current_yes_price=mkt["yes_price"],
            days_to_end=None,
            rag_context="",
            web_context=web_ctx,
        )

        # Gemma3:4b
        print("      Calling gemma3:4b ...", end="", flush=True)
        gemma_result = await call_ollama(prompt, cfg)
        gemma_prob = gemma_result.get("probability_estimate", -1) if gemma_result else -1
        gemma_conf = gemma_result.get("confidence", "?") if gemma_result else "?"
        print(f" prob={gemma_prob:.3f} conf={gemma_conf}" if gemma_prob >= 0 else " FAILED")

        # Claude
        print("      Calling Claude ...", end="", flush=True)
        claude_result = await call_claude(prompt, cfg)
        claude_prob = claude_result.get("probability_estimate", -1) if claude_result else -1
        claude_conf = claude_result.get("confidence", "?") if claude_result else "?"
        print(f" prob={claude_prob:.3f} conf={claude_conf}" if claude_prob >= 0 else " FAILED")

        results.append({
            "question": q[:50],
            "market": mkt["yes_price"],
            "gemma": gemma_prob,
            "claude": claude_prob,
            "gemma_conf": gemma_conf,
            "claude_conf": claude_conf,
            "d_gemma": abs(gemma_prob - mkt["yes_price"]) if gemma_prob >= 0 else None,
            "d_claude": abs(claude_prob - mkt["yes_price"]) if claude_prob >= 0 else None,
        })
        print()

    # Summary
    print("=" * 78)
    print(f"  {'Market':<50}  {'Price':>5}  {'Gemma':>5}  {'Claude':>6}  {'Winner':>7}")
    print("-" * 78)
    claude_wins = 0
    gemma_wins = 0
    for r in results:
        if r["d_gemma"] is not None and r["d_claude"] is not None:
            winner = "Claude" if r["d_claude"] <= r["d_gemma"] else "Gemma"
            if winner == "Claude":
                claude_wins += 1
            else:
                gemma_wins += 1
        else:
            winner = "N/A"
        g = f"{r['gemma']:.3f}" if r["gemma"] >= 0 else "FAIL"
        c = f"{r['claude']:.3f}" if r["claude"] >= 0 else "FAIL"
        print(f"  {r['question']:<50}  {r['market']:>5.3f}  {g:>5}  {c:>6}  {winner:>7}")

    print("-" * 78)
    total = len([r for r in results if r["d_gemma"] is not None and r["d_claude"] is not None])
    print(f"  Claude closer to market: {claude_wins}/{total}")
    print(f"  Gemma closer to market:  {gemma_wins}/{total}")

    # Average deviations
    g_devs = [r["d_gemma"] for r in results if r["d_gemma"] is not None]
    c_devs = [r["d_claude"] for r in results if r["d_claude"] is not None]
    if g_devs:
        print(f"\n  Gemma avg deviation from market:  {sum(g_devs)/len(g_devs):.3f}")
    if c_devs:
        print(f"  Claude avg deviation from market: {sum(c_devs)/len(c_devs):.3f}")
    print("=" * 78)

    await searcher.close()


if __name__ == "__main__":
    asyncio.run(main())
