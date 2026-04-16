from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.main import _select_independent_mm_markets
from src.polymarket.models import Market, MarketToken


def _end_date_in_minutes(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def _market(
    cid: str,
    *,
    minutes_to_end: int,
    volume_24h: float,
    liquidity: float,
    yes_price: float = 0.55,
) -> Market:
    return Market(
        condition_id=cid,
        question_id=f"qid-{cid[-4:]}",
        question=f"Test market {cid[-4:]}?",
        tokens=[
            MarketToken(token_id=f"yes-{cid[-4:]}", outcome="Yes", price=yes_price),
            MarketToken(token_id=f"no-{cid[-4:]}", outcome="No", price=1 - yes_price),
        ],
        volume=1_000_000,
        volume_24hr=volume_24h,
        liquidity=liquidity,
        end_date_iso=_end_date_in_minutes(minutes_to_end),
        active=True,
        closed=False,
        archived=False,
    )


def test_short_horizon_selector_blocks_long_horizon_and_low_quality(settings):
    settings.mm_independent_discovery_enabled = True
    settings.mm_short_horizon_min_minutes = 5
    settings.mm_short_horizon_max_hours = 6
    settings.mm_min_market_volume_24h_usd = 100_000
    settings.mm_min_market_liquidity_usd = 100_000
    settings.mm_independent_max_markets = 20

    eligible = _market(
        "0x" + "1" * 64,
        minutes_to_end=30,
        volume_24h=200_000,
        liquidity=250_000,
    )
    long_horizon = _market(
        "0x" + "2" * 64,
        minutes_to_end=60 * 24 * 3,
        volume_24h=300_000,
        liquidity=300_000,
    )
    low_liquidity = _market(
        "0x" + "3" * 64,
        minutes_to_end=45,
        volume_24h=300_000,
        liquidity=50_000,
    )

    selected = _select_independent_mm_markets(
        {
            eligible.condition_id: eligible,
            long_horizon.condition_id: long_horizon,
            low_liquidity.condition_id: low_liquidity,
        },
        settings,
    )

    assert list(selected.keys()) == [eligible.condition_id]


def test_short_horizon_selector_respects_limit_and_priority(settings):
    settings.mm_independent_discovery_enabled = True
    settings.mm_short_horizon_min_minutes = 5
    settings.mm_short_horizon_max_hours = 12
    settings.mm_min_market_volume_24h_usd = 50_000
    settings.mm_min_market_liquidity_usd = 50_000
    settings.mm_independent_max_markets = 1

    high_volume = _market(
        "0x" + "4" * 64,
        minutes_to_end=90,
        volume_24h=900_000,
        liquidity=500_000,
    )
    lower_volume = _market(
        "0x" + "5" * 64,
        minutes_to_end=90,
        volume_24h=120_000,
        liquidity=600_000,
    )

    selected = _select_independent_mm_markets(
        {
            lower_volume.condition_id: lower_volume,
            high_volume.condition_id: high_volume,
        },
        settings,
    )

    assert list(selected.keys()) == [high_volume.condition_id]
