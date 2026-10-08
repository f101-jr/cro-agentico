"""Shopify: filtro de pedidos, buckets y métricas derivadas."""

from __future__ import annotations

from connectors.base import FetchRequest
from connectors.shopify import ORDER_FILTER, bucket_of
from connectors.windows import day_offsets
from fixtures.dataset import load_dataset
from semantic.schema import Device, Window


async def test_orders_match_the_dataset(connectors):
    _, shopify, _ = connectors
    expected = load_dataset().shopify_totals(day_offsets(Window.LAST_14D), "mobile")
    obs = await shopify.fetch(FetchRequest(metric="orders_summary", window=Window.LAST_14D,
                                           device=Device.MOBILE))
    values = {o.metric: o.value for o in obs}
    assert values["orders"] == expected["orders"]
    assert values["revenue"] == expected["revenue_eur"]
    assert values["aov"] == round(expected["revenue_eur"] / expected["orders"], 2)


async def test_filter_is_documented_in_the_note(connectors):
    _, shopify, _ = connectors
    obs = await shopify.fetch(FetchRequest(metric="orders_summary", window=Window.LAST_7D))
    assert all(ORDER_FILTER in (o.note or "") for o in obs)


async def test_bucket_breakdown_sums_to_total(connectors):
    _, shopify, _ = connectors
    total = await shopify.fetch(FetchRequest(metric="orders_summary", window=Window.LAST_14D))
    buckets = await shopify.fetch(FetchRequest(metric="orders_summary", window=Window.LAST_14D,
                                               dimension="aov_bucket"))
    total_orders = next(o.value for o in total if o.metric == "orders")
    bucket_orders = sum(o.value for o in buckets if o.metric == "orders")
    assert bucket_orders == total_orders


async def test_cancelled_orders_are_excluded(connectors):
    from connectors.shopify import ShopifyConnector

    class OneCancelled:
        async def iter_orders(self, **kwargs):
            yield {"id": "1", "cancelled_at": None, "financial_status": "paid", "device": "mobile",
                   "channel": "direct", "current_total_price": "100.00", "total_refunded": "0",
                   "line_items": []}
            yield {"id": "2", "cancelled_at": "2026-01-01", "financial_status": "paid",
                   "device": "mobile", "channel": "direct", "current_total_price": "900.00",
                   "total_refunded": "0", "line_items": []}

        async def ping(self):
            return None

    connector = ShopifyConnector(OneCancelled(), retry_delay=0.0)
    obs = await connector.fetch(FetchRequest(metric="orders_summary", window=Window.LAST_7D))
    values = {o.metric: o.value for o in obs}
    assert values["orders"] == 1 and values["revenue"] == 100.0


def test_bucket_boundaries():
    assert bucket_of(49.99) == "lt_50"
    assert bucket_of(50.0) == "50_75" and bucket_of(75.0) == "50_75"
    assert bucket_of(75.01) == "gt_75"
