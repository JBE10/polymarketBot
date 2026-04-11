from __future__ import annotations

import asyncio

import pytest

from src.polymarket.clob_client import AsyncClobClient
from src.polymarket.models import OrderRequest, OrderType, Side


class _FakeSdkClient:
    def __init__(self) -> None:
        self.post_calls = 0
        self.failures_before_success = 0
        self.failure_message = ""

    def create_order(self, order_args):
        return {"signed": True, "args": order_args}

    def post_order(self, order, order_type):
        self.post_calls += 1
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError(self.failure_message)
        return {
            "orderID": f"oid-{self.post_calls}",
            "status": "MATCHED",
            "sizeMatched": "10",
        }


@pytest.mark.asyncio
async def test_idempotency_key_deduplicates_repeated_submission() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    fake = _FakeSdkClient()
    client._client = fake

    req = OrderRequest(
        token_id="tok-1",
        price=0.55,
        size=10,
        side=Side.BUY,
        order_type=OrderType.GTC,
    )

    first = await client.place_order(req, idempotency_key="k1", max_retries=0, base_backoff_seconds=0)
    second = await client.place_order(req, idempotency_key="k1", max_retries=0, base_backoff_seconds=0)

    assert first.order_id == second.order_id
    assert fake.post_calls == 1

    await client.close()


@pytest.mark.asyncio
async def test_idempotency_lock_deduplicates_concurrent_submission() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    fake = _FakeSdkClient()
    client._client = fake

    req = OrderRequest(
        token_id="tok-2",
        price=0.56,
        size=12,
        side=Side.SELL,
        order_type=OrderType.GTC,
    )

    r1, r2 = await asyncio.gather(
        client.place_order(req, idempotency_key="k2", max_retries=0, base_backoff_seconds=0),
        client.place_order(req, idempotency_key="k2", max_retries=0, base_backoff_seconds=0),
    )

    assert r1.order_id == r2.order_id
    assert fake.post_calls == 1

    await client.close()


@pytest.mark.asyncio
async def test_retries_transient_error_then_succeeds() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    fake = _FakeSdkClient()
    fake.failures_before_success = 2
    fake.failure_message = "429 too many requests"
    client._client = fake

    req = OrderRequest(
        token_id="tok-3",
        price=0.57,
        size=10,
        side=Side.BUY,
        order_type=OrderType.GTC,
    )

    resp = await client.place_order(req, max_retries=3, base_backoff_seconds=0)

    assert resp.success is True
    assert fake.post_calls == 3

    await client.close()


@pytest.mark.asyncio
async def test_non_transient_error_does_not_retry() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    fake = _FakeSdkClient()
    fake.failures_before_success = 1
    fake.failure_message = "invalid signature"
    client._client = fake

    req = OrderRequest(
        token_id="tok-4",
        price=0.58,
        size=10,
        side=Side.BUY,
        order_type=OrderType.GTC,
    )

    resp = await client.place_order(req, max_retries=3, base_backoff_seconds=0)

    assert resp.success is False
    assert resp.status.value == "REJECTED"
    assert fake.post_calls == 1

    await client.close()
