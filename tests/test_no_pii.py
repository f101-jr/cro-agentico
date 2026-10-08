"""La PII muere en el mapeo del conector, antes de cualquier log."""

from __future__ import annotations

from connectors.base import FetchRequest, contains_pii
from semantic.schema import Device, Window


async def _serialized(connector, **kwargs):
    obs = await connector.fetch(FetchRequest(**kwargs))
    return [o.model_dump(mode="json") for o in obs]


async def test_no_pii_in_observations(connectors):
    ga4, shopify, support = connectors
    payloads: list = []
    for window in (Window.LAST_7D, Window.LAST_14D):
        payloads += await _serialized(ga4, metric="sessions", window=window, device=Device.MOBILE)
        payloads += await _serialized(ga4, metric="funnel_steps", window=window)
        payloads += await _serialized(ga4, metric="traffic_by_dim", window=window, dimension="country")
        payloads += await _serialized(shopify, metric="orders_summary", window=window, dimension="channel")
        payloads += await _serialized(shopify, metric="product_performance", window=window)
        payloads += await _serialized(shopify, metric="refunds", window=window)
        payloads += await _serialized(support, metric="support_tickets", window=window)
    assert payloads
    found = contains_pii(payloads)
    assert found == [], f"PII en la salida de un conector: {found[:5]}"


async def test_transport_really_carries_pii(shopify_transport):
    """Si el transporte no trajera PII, el test anterior no probaría nada."""
    from datetime import date, timedelta

    today = date.today()
    orders = [o async for o in shopify_transport.iter_orders(
        start=today - timedelta(days=2), end=today - timedelta(days=1), device=Device.MOBILE)]
    assert orders, "el transporte de fixtures no ha devuelto pedidos"
    assert contains_pii(orders[:5]), "el doble de test debe traer PII para que el mapeo la descarte"


def test_strip_pii_removes_fields_and_patterns():
    from connectors.base import strip_pii

    dirty = {"customer": {"email": "a@b.es"}, "note": "escribe a cliente@tienda.es",
             "phone": "+34 600 123 456", "ok": "sin datos"}
    clean = strip_pii(dirty)
    assert "customer" not in clean and "phone" not in clean
    assert clean["note"] == "[redacted]"
    assert clean["ok"] == "sin datos"
