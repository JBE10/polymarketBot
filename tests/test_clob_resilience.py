from __future__ import annotations

import asyncio
from typing import Any

import httpx
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


class _FakeHttpClient:
    def __init__(self, sequence: list[Any]) -> None:
        self._sequence = sequence
        self.calls = 0

    async def get(self, _path: str, params=None):
        self.calls += 1
        item = self._sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _http_response(status: int, payload: Any, url: str = "https://example.test") -> httpx.Response:
    req = httpx.Request("GET", url)
    return httpx.Response(status, json=payload, request=req)


def _http_status_error(status: int, payload: Any, url: str = "https://example.test") -> httpx.HTTPStatusError:
    response = _http_response(status, payload, url=url)
    request = response.request
    return httpx.HTTPStatusError("status error", request=request, response=response)


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


@pytest.mark.asyncio
async def test_get_order_book_retries_timeout_then_succeeds() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    client._http = _FakeHttpClient(
        [
            httpx.TimeoutException("timed out"),
            _http_response(
                200,
                {
                    "market": "m",
                    "asset_id": "tok",
                    "bids": [{"price": "0.61", "size": "100"}],
                    "asks": [{"price": "0.63", "size": "100"}],
                },
            ),
        ]
    )

    book = await client.get_order_book("tok")

    assert book is not None
    assert book.best_bid == 0.61
    assert client._http.calls == 2


@pytest.mark.asyncio
async def test_get_markets_retries_429_then_succeeds() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    client._gamma = _FakeHttpClient(
        [
            _http_status_error(429, {"error": "too many requests"}),
            _http_response(
                200,
                [
                    {
                        "conditionId": "0x" + "c" * 64,
                        "questionID": "qid_1",
                        "question": "Q?",
                        "clobTokenIds": '["yes_tok", "no_tok"]',
                        "outcomes": ["Yes", "No"],
                        "outcomePrices": '["0.60", "0.40"]',
                        "active": True,
                        "closed": False,
                        "archived": False,
                    }
                ],
            ),
        ]
    )

    markets = await client.get_markets(limit=1)

    assert len(markets) == 1
    assert markets[0].condition_id.startswith("0x")
    assert client._gamma.calls == 2


@pytest.mark.asyncio
async def test_get_price_does_not_retry_non_transient_400() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    client._http = _FakeHttpClient(
        [_http_status_error(400, {"error": "bad request"})]
    )

    price = await client.get_price("tok", Side.BUY)

    assert price is None
    assert client._http.calls == 1


@pytest.mark.asyncio
async def test_get_trades_dry_run_skips_clob_and_uses_gamma() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    client._http = _FakeHttpClient([])
    client._gamma = _FakeHttpClient(
        [
            _http_response(
                200,
                {
                    "history": [
                        {"t": 1, "p": 0.61},
                        {"t": 2, "p": 0.62},
                    ]
                },
            )
        ]
    )

    trades = await client.get_trades("tok", limit=2)

    assert len(trades) == 2
    assert client._http.calls == 0
    assert client._gamma.calls == 1


@pytest.mark.asyncio
async def test_get_order_book_fallbacks_to_asset_id_on_404() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    client._http = _FakeHttpClient(
        [
            _http_response(404, {"error": "not found"}),
            _http_response(
                200,
                {
                    "market": "m",
                    "asset_id": "tok",
                    "bids": [{"price": "0.51", "size": "100"}],
                    "asks": [{"price": "0.53", "size": "100"}],
                },
            ),
        ]
    )

    book = await client.get_order_book("tok")

    assert book is not None
    assert book.best_bid == 0.51
    assert client._http.calls == 2


@pytest.mark.asyncio
async def test_get_trades_gamma_404_is_cached() -> None:
    client = AsyncClobClient(host="https://clob.polymarket.com", private_key="0xabc", dry_run=True)
    client._gamma = _FakeHttpClient([
        _http_response(404, {"error": "not found"}),
    ])

    trades_first = await client.get_trades("tok404", limit=5)
    trades_second = await client.get_trades("tok404", limit=5)

    assert trades_first == []
    assert trades_second == []
    assert client._gamma.calls == 1
