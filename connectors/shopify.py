"""Conector Shopify.

Fuente de verdad de pedidos, ingresos, AOV y devoluciones.

Filtro aplicado a `orders` (se propaga literal a `Observation.note`):
    cancelled_at IS NULL AND financial_status NOT IN (voided)
    y el revenue se descuenta de las devoluciones (neto).

PII: el transporte devuelve pedidos con datos de cliente — igual que la API
real — y el conector los **descarta en el mapeo**, antes de cualquier log.
Nada de lo que sale de `fetch()` contiene un campo de cliente.

Scopes requeridos (solo lectura): `read_orders`, `read_products`.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator, Iterable, Protocol, runtime_checkable

from connectors.base import (
    Capability,
    FetchRequest,
    HealthStatus,
    TransientSourceError,
    strip_pii,
    with_retries,
)
from semantic.schema import DataQuality, Device, Observation, Window

SOURCE = "shopify"

ORDER_FILTER = ("cancelled_at IS NULL, financial_status != voided, "
                "revenue neto de devoluciones (total - refunds)")
BUCKETS = ("lt_50", "50_75", "gt_75")
BUCKET_LABEL = {"lt_50": "<50€", "50_75": "50-75€", "gt_75": ">75€"}
DIMENSIONS = ("none", "device", "channel", "product_type", "aov_bucket")

PRODUCT_TYPE_NOTE = ("Cada pedido se asigna a la categoría de su línea de mayor importe, "
                     "de modo que la suma por categoría cuadra con el total de pedidos.")
PRODUCT_LINE_NOTE = ("Importes a nivel de línea de producto: no cuadran con el revenue del "
                     "pedido porque envío y descuentos van a nivel de pedido.")
PRODUCT_RATE_NOTE = ("Proporción de pedidos que incluyen el producto, no una conversión por "
                     "sesión: Shopify no ve las sesiones.")


def bucket_of(amount: float) -> str:
    if amount < 50:
        return "lt_50"
    if amount <= 75:
        return "50_75"
    return "gt_75"


@runtime_checkable
class ShopifyTransport(Protocol):
    async def iter_orders(self, *, start: date, end: date, device: Device) -> AsyncIterator[dict]: ...
    async def ping(self) -> date | None: ...


# --------------------------------------------------------------------------- #
# Transporte: fixtures (sintetiza pedidos a partir de los agregados diarios)
# --------------------------------------------------------------------------- #
PHI = 0.6180339887498949
REFUND_EVERY = 29  # 1 de cada 29 pedidos lleva devolución parcial


class ShopifyFixtureTransport:
    """Reconstruye pedidos individuales — con cliente ficticio — desde `fixtures/`."""

    def __init__(self, *, timezone_name: str = "Europe/Madrid") -> None:
        from fixtures.dataset import load_dataset

        self._ds = load_dataset()
        self.timezone_name = timezone_name
        self.calls = 0

    async def iter_orders(self, *, start: date, end: date, device: Device) -> AsyncIterator[dict]:
        self.calls += 1
        from fixtures.dataset import allocate

        today = _today(self.timezone_name)
        offsets = list(range((today - end).days, (today - start).days + 1))
        catalog = list(self._ds.product_catalog().values())
        seq = 0
        for row in self._ds.shopify_daily(offsets, device.value):
            offset = row["day_offset"]
            created = today - timedelta(days=offset)
            dev = row["device"]
            channels = allocate(row["orders"], self._ds.channel_order_shares(dev))
            channel_seq = [c for c, n in channels.items() for _ in range(n)]
            post = offset <= 7
            idx = 0
            for bucket in BUCKETS:
                n = row["buckets"][bucket]["orders"]
                if not n:
                    continue
                prices = _split_cents(row["buckets"][bucket]["revenue_eur"], n)
                for i in range(n):
                    seq += 1
                    net = prices[i]
                    refunded = round(net * 0.5, 2) if seq % REFUND_EVERY == 0 else 0.0
                    yield {
                        "id": f"gid://shopify/Order/{offset:03d}{dev[:3]}{seq:05d}",
                        "created_at": datetime(created.year, created.month, created.day,
                                               (seq * 7) % 24, (seq * 13) % 60).isoformat(),
                        "cancelled_at": None,
                        "financial_status": "paid",
                        "device": dev,
                        "channel": channel_seq[idx % len(channel_seq)] if channel_seq else "direct",
                        "current_total_price": f"{net + refunded:.2f}",
                        "total_refunded": f"{refunded:.2f}",
                        # --- PII que el conector DEBE descartar ---------------
                        "customer": {
                            "id": f"cust-{seq}",
                            "email": f"cliente{seq}@ejemplo.es",
                            "first_name": "Cliente",
                            "last_name": f"Demo{seq}",
                            "phone": f"+34 6{seq % 10}{seq % 100:02d} {seq % 1000:03d} {seq % 100:02d}",
                        },
                        "shipping_address": {"city": "Zaragoza", "zip": "50001",
                                             "address1": f"Calle Falsa {seq}"},
                        "line_items": _line_items(catalog, seq, post, net),
                    }
                    idx += 1

    async def ping(self) -> date | None:
        self.calls += 1
        return _today(self.timezone_name) - timedelta(days=1)


def _split_cents(total_eur: float, n: int) -> list[float]:
    cents = round(total_eur * 100)
    base, rest = divmod(cents, n)
    return [(base + (1 if i < rest else 0)) / 100 for i in range(n)]


def _line_items(catalog: list[dict], seq: int, post: bool, order_total: float) -> list[dict]:
    """1-2 líneas por pedido, elegidas con una secuencia de baja discrepancia."""
    key = "unit_share_post" if post else "unit_share_pre"
    n_lines = 2 if (seq % 5) in (0, 1, 2) else 1  # ~1,6 líneas por pedido
    lines = []
    for j in range(n_lines):
        u = (seq * PHI + j * 0.37) % 1.0
        acc = 0.0
        chosen = catalog[-1]
        for p in catalog:
            acc += p[key]
            if u <= acc:
                chosen = p
                break
        lines.append({
            "product_id": chosen["id"],
            "title": chosen["title"],
            "product_type": chosen["product_type"],
            "quantity": 1,
            "price": f"{chosen['price_eur']:.2f}",
        })
    return lines


# --------------------------------------------------------------------------- #
# Transporte: Admin GraphQL API
# --------------------------------------------------------------------------- #
ORDERS_QUERY = """
query Orders($query: String!, $cursor: String) {
  orders(first: 250, after: $cursor, query: $query, sortKey: CREATED_AT) {
    pageInfo { hasNextPage endCursor }
    edges { node {
      id createdAt cancelledAt displayFinancialStatus
      currentTotalPriceSet { shopMoney { amount } }
      totalRefundedSet { shopMoney { amount } }
      customerJourneySummary { firstVisit { source } }
      lineItems(first: 50) { edges { node {
        quantity
        product { id productType title }
        originalUnitPriceSet { shopMoney { amount } }
      } } }
    } }
  }
}
"""


class ShopifyGraphQLTransport:
    """Admin GraphQL API con paginación por cursor hasta agotar la ventana."""

    def __init__(self, shop: str | None = None, token: str | None = None,
                 *, api_version: str | None = None, client=None) -> None:
        self.shop = shop or os.environ["CRO_SHOPIFY_SHOP"]
        self.token = token or os.environ["CRO_SHOPIFY_TOKEN"]
        self.api_version = api_version or os.getenv("CRO_SHOPIFY_API_VERSION", "2024-10")
        self._client = client

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=f"https://{self.shop}/admin/api/{self.api_version}",
                headers={"X-Shopify-Access-Token": self.token},
                timeout=20.0,
            )
        return self._client

    async def iter_orders(self, *, start: date, end: date, device: Device) -> AsyncIterator[dict]:
        query = (f"created_at:>={start.isoformat()} created_at:<={end.isoformat()} "
                 f"status:any -status:cancelled")
        cursor = None
        while True:
            payload = await self._post({"query": ORDERS_QUERY, "variables": {"query": query, "cursor": cursor}})
            orders = payload["data"]["orders"]
            for edge in orders["edges"]:
                node = edge["node"]
                order = _from_graphql(node)
                if device is not Device.ALL and order["device"] != device.value:
                    continue
                yield order
            if not orders["pageInfo"]["hasNextPage"]:
                return
            cursor = orders["pageInfo"]["endCursor"]

    async def _post(self, body: dict) -> dict:
        response = await self._http().post("/graphql.json", json=body)
        if response.status_code in (429, 503):
            raise TransientSourceError(SOURCE, response.status_code, response.text[:200])
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"shopify graphql: {payload['errors']}")
        return payload

    async def ping(self) -> date | None:
        await self._post({"query": "{ shop { name } }"})
        return _today() - timedelta(days=0)


def _from_graphql(node: dict) -> dict:
    lines = [
        {
            "product_id": (e["node"].get("product") or {}).get("id"),
            "title": (e["node"].get("product") or {}).get("title"),
            "product_type": (e["node"].get("product") or {}).get("productType") or "unknown",
            "quantity": e["node"]["quantity"],
            "price": e["node"]["originalUnitPriceSet"]["shopMoney"]["amount"],
        }
        for e in node["lineItems"]["edges"]
    ]
    journey = (node.get("customerJourneySummary") or {}).get("firstVisit") or {}
    return {
        "id": node["id"],
        "created_at": node["createdAt"],
        "cancelled_at": node.get("cancelledAt"),
        "financial_status": node.get("displayFinancialStatus", ""),
        # Shopify no expone deviceCategory: se deduce del canal de origen.
        "device": node.get("device", "unknown"),
        "channel": journey.get("source") or "direct",
        "current_total_price": node["currentTotalPriceSet"]["shopMoney"]["amount"],
        "total_refunded": (node.get("totalRefundedSet") or {}).get("shopMoney", {}).get("amount", "0"),
        "line_items": lines,
    }


# --------------------------------------------------------------------------- #
# Conector
# --------------------------------------------------------------------------- #
class ShopifyConnector:
    name = SOURCE

    def __init__(self, transport: ShopifyTransport, *, retry_delay: float = 0.5,
                 timezone_name: str = "Europe/Madrid") -> None:
        self.transport = transport
        self._retry_delay = retry_delay
        self.timezone_name = timezone_name

    def capabilities(self) -> list[Capability]:
        windows = list(Window)
        return [
            Capability(metric="orders", dimensions=list(DIMENSIONS), windows=windows),
            Capability(metric="revenue", dimensions=list(DIMENSIONS), windows=windows),
            Capability(metric="aov", dimensions=list(DIMENSIONS), windows=windows),
            Capability(metric="refunds", dimensions=["none", "device"], windows=windows),
            Capability(metric="product_performance", dimensions=["product"], windows=windows),
        ]

    def window_dates(self, window: Window) -> tuple[date, date]:
        from zoneinfo import ZoneInfo

        from connectors.windows import resolve_window

        return resolve_window(window, tz=ZoneInfo(self.timezone_name))

    async def fetch(self, request: FetchRequest) -> list[Observation]:
        start, end = self.window_dates(request.window)

        async def _collect() -> list[dict]:
            # El descarte de PII ocurre AQUÍ, en el mapeo, antes de cualquier log.
            return [
                _clean_order(o)
                async for o in self.transport.iter_orders(start=start, end=end, device=request.device)
                if _is_countable(o)
            ]

        orders = await with_retries(_collect, source=SOURCE, base_delay=self._retry_delay)
        base = dict(window=request.window, window_start=start, window_end=end, source=SOURCE,
                    quality=DataQuality.EXACT, fetched_at=datetime.now(timezone.utc))
        device_dim = {"device": request.device.value}

        if request.metric == "product_performance":
            return self._product_observations(orders, base, device_dim, request.limit or 100)
        if request.metric == "refunds":
            return self._refund_observations(orders, base, device_dim)
        return self._summary_observations(orders, base, device_dim, request.dimension or "none")

    # ---------------------------------------------------------------- mapeo --
    def _summary_observations(self, orders, base, device_dim, dimension) -> list[Observation]:
        if dimension not in DIMENSIONS:
            raise ValueError(f"dimensión no soportada por shopify: {dimension}")
        groups: dict[str | None, list[dict]] = {}
        for order in orders:
            groups.setdefault(_group_key(order, dimension), []).append(order)

        note = f"Filtro aplicado: {ORDER_FILTER}."
        if dimension == "product_type":
            note += " " + PRODUCT_TYPE_NOTE
        obs: list[Observation] = []
        for key in sorted(groups, key=lambda k: (-len(groups[k]), str(k))):
            group = groups[key]
            dims = dict(device_dim)
            if dimension != "none":
                dims[dimension] = str(key)
            count = len(group)
            revenue = round(sum(o["net_total"] for o in group), 2)
            obs.append(Observation(metric="orders", value=float(count), unit="count",
                                   dimensions=dims, sample_size=count, note=note, **base))
            obs.append(Observation(metric="revenue", value=revenue, unit="currency_eur",
                                   dimensions=dims, sample_size=count, note=note, **base))
            obs.append(Observation(metric="aov", value=round(revenue / count, 2) if count else 0.0,
                                   unit="currency_eur", dimensions=dims, sample_size=count,
                                   note=note + " AOV = revenue / orders, calculado en Python.", **base))
        return obs

    def _refund_observations(self, orders, base, device_dim) -> list[Observation]:
        refunded = [o for o in orders if o["refunded"] > 0]
        total = round(sum(o["refunded"] for o in orders), 2)
        return [
            Observation(metric="refunds", value=total, unit="currency_eur", dimensions=dict(device_dim),
                        sample_size=len(orders), note=f"Importe devuelto. Filtro: {ORDER_FILTER}.", **base),
            Observation(metric="refunded_orders", value=float(len(refunded)), unit="count",
                        dimensions=dict(device_dim), sample_size=len(orders),
                        note="Pedidos con alguna devolución.", **base),
        ]

    def _product_observations(self, orders, base, device_dim, limit) -> list[Observation]:
        units: dict[str, int] = {}
        revenue: dict[str, float] = {}
        orders_with: dict[str, int] = {}
        titles: dict[str, str] = {}
        for order in orders:
            seen = set()
            for line in order["lines"]:
                pid = line["product_id"] or "unknown"
                titles[pid] = line["title"] or pid
                units[pid] = units.get(pid, 0) + line["quantity"]
                revenue[pid] = round(revenue.get(pid, 0.0) + line["quantity"] * line["price"], 2)
                seen.add(pid)
            for pid in seen:
                orders_with[pid] = orders_with.get(pid, 0) + 1

        total_orders = len(orders)
        ranked = sorted(units, key=lambda p: (-units[p], p))[:limit]
        obs: list[Observation] = []
        for pid in ranked:
            dims = {**device_dim, "product": titles.get(pid, pid)}
            obs.append(Observation(metric="product_units", value=float(units[pid]), unit="count",
                                   dimensions=dims, sample_size=units[pid], note=PRODUCT_LINE_NOTE, **base))
            obs.append(Observation(metric="product_revenue", value=revenue[pid], unit="currency_eur",
                                   dimensions=dims, sample_size=units[pid], note=PRODUCT_LINE_NOTE, **base))
            rate = (orders_with.get(pid, 0) / total_orders) if total_orders else 0.0
            obs.append(Observation(metric="product_order_rate", value=round(rate, 6), unit="ratio",
                                   dimensions=dims, sample_size=total_orders, note=PRODUCT_RATE_NOTE, **base))
        return obs

    async def health(self) -> HealthStatus:
        started = time.perf_counter()
        try:
            last_complete = await with_retries(self.transport.ping, source=SOURCE, base_delay=self._retry_delay)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(source=SOURCE, reachable=False,
                                latency_ms=int((time.perf_counter() - started) * 1000), message=str(exc))
        return HealthStatus(source=SOURCE, reachable=True, last_complete_date=last_complete,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            message=f"pedidos con filtro: {ORDER_FILTER}")


def _is_countable(order: dict) -> bool:
    """Excluye cancelados y anulados."""
    return not order.get("cancelled_at") and str(order.get("financial_status", "")).lower() not in {"voided", "refunded"}


def _clean_order(order: dict) -> dict:
    """Mapeo a la forma interna. Aquí muere la PII: nada de cliente sobrevive."""
    safe = strip_pii(order)
    total = float(safe.get("current_total_price") or 0.0)
    refunded = float(safe.get("total_refunded") or 0.0)
    lines = [
        {
            "product_id": line.get("product_id"),
            "title": line.get("title"),
            "product_type": line.get("product_type") or "unknown",
            "quantity": int(line.get("quantity") or 0),
            "price": float(line.get("price") or 0.0),
        }
        for line in safe.get("line_items", [])
    ]
    return {
        "device": safe.get("device", "unknown"),
        "channel": safe.get("channel", "direct"),
        "net_total": round(total - refunded, 2),
        "refunded": refunded,
        "lines": lines,
    }


def _group_key(order: dict, dimension: str) -> str:
    if dimension == "none":
        return "all"
    if dimension == "device":
        return order["device"]
    if dimension == "channel":
        return order["channel"]
    if dimension == "aov_bucket":
        return BUCKET_LABEL[bucket_of(order["net_total"])]
    dominant = max(order["lines"], key=lambda line: line["quantity"] * line["price"], default=None)
    return dominant["product_type"] if dominant else "unknown"


def _today(tz_name: str = "Europe/Madrid") -> date:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(tz_name)).date()


def _iter(values: Iterable[dict]) -> list[dict]:
    return list(values)


__all__ = [
    "ShopifyConnector",
    "ShopifyFixtureTransport",
    "ShopifyGraphQLTransport",
    "ShopifyTransport",
    "bucket_of",
]
