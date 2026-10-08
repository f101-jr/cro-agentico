"""Acceso tipado al dataset de fixtures.

Lo consumen **solo** las capas de transporte de los conectores: la
capa semántica y el orquestador nunca importan este módulo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEVICES = ("mobile", "desktop", "tablet")
BUCKETS = ("lt_50", "50_75", "gt_75")
CHANGE_LIVE_FROM_OFFSET = 7  # offsets 1..7 con el umbral nuevo


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FixtureDataset:
    meta: dict
    ga4_meta: dict
    ga4_rows: list[dict]
    ga4_shares: dict
    shopify_meta: dict
    shopify_rows: list[dict]
    shopify_shares: dict
    products: dict
    tickets: list[dict]

    # ---------------------------------------------------------------- GA4 ---
    def ga4_daily(self, offsets: Iterable[int], device: str = "all") -> list[dict]:
        offs = set(offsets)
        return [
            r for r in self.ga4_rows
            if r["day_offset"] in offs and (device == "all" or r["device"] == device)
        ]

    def ga4_totals(self, offsets: Iterable[int], device: str = "all") -> dict[str, int]:
        keys = ("sessions", "users", "views", "carts", "checkouts", "purchases")
        out = dict.fromkeys(keys, 0)
        for r in self.ga4_daily(offsets, device):
            for k in keys:
                out[k] += r[k]
        return out

    def ga4_by_device(self, offsets: Iterable[int]) -> dict[str, dict[str, int]]:
        return {d: self.ga4_totals(offsets, d) for d in DEVICES}

    def dimension_shares(self, dimension: str, device: str) -> dict[str, float]:
        """Cuotas de sesiones por valor de dimensión. `device='all'` las mezcla."""
        table = self.ga4_shares[dimension]
        if device != "all":
            return dict(table[device])
        weights = {d: self.ga4_totals(range(1, 31), d)["sessions"] for d in DEVICES}
        total = sum(weights.values()) or 1
        merged: dict[str, float] = {}
        for d in DEVICES:
            for value, share in table[d].items():
                merged[value] = merged.get(value, 0.0) + share * weights[d] / total
        return merged

    def channel_cr_index(self) -> dict[str, float]:
        return dict(self.ga4_shares["channel_cr_index"])

    # ------------------------------------------------------------ Shopify ---
    def shopify_daily(self, offsets: Iterable[int], device: str = "all") -> list[dict]:
        offs = set(offsets)
        return [
            r for r in self.shopify_rows
            if r["day_offset"] in offs and (device == "all" or r["device"] == device)
        ]

    def shopify_totals(self, offsets: Iterable[int], device: str = "all") -> dict:
        orders = 0
        revenue = 0.0
        buckets = {b: {"orders": 0, "revenue_eur": 0.0} for b in BUCKETS}
        for r in self.shopify_daily(offsets, device):
            orders += r["orders"]
            revenue += r["revenue_eur"]
            for b in BUCKETS:
                buckets[b]["orders"] += r["buckets"][b]["orders"]
                buckets[b]["revenue_eur"] += r["buckets"][b]["revenue_eur"]
        buckets = {b: {"orders": v["orders"], "revenue_eur": round(v["revenue_eur"], 2)}
                   for b, v in buckets.items()}
        return {"orders": orders, "revenue_eur": round(revenue, 2), "buckets": buckets}

    def shopify_by_device(self, offsets: Iterable[int]) -> dict[str, dict]:
        return {d: self.shopify_totals(offsets, d) for d in DEVICES}

    def channel_order_shares(self, device: str) -> dict[str, float]:
        table = self.shopify_shares["channel_order_shares"]
        if device != "all":
            return dict(table[device])
        weights = {d: self.shopify_totals(range(1, 31), d)["orders"] for d in DEVICES}
        total = sum(weights.values()) or 1
        merged: dict[str, float] = {}
        for d in DEVICES:
            for value, share in table[d].items():
                merged[value] = merged.get(value, 0.0) + share * weights[d] / total
        return merged

    def post_change_ratio(self, offsets: Iterable[int]) -> float:
        """Fracción de días de la ventana con el umbral nuevo ya activo."""
        offs = list(offsets)
        if not offs:
            return 0.0
        return sum(1 for o in offs if o <= CHANGE_LIVE_FROM_OFFSET) / len(offs)

    def product_unit_shares(self, offsets: Iterable[int]) -> dict[str, float]:
        w = self.post_change_ratio(offsets)
        shares = {
            p["id"]: p["unit_share_pre"] * (1 - w) + p["unit_share_post"] * w
            for p in self.products["products"]
        }
        total = sum(shares.values()) or 1.0
        return {k: v / total for k, v in shares.items()}

    def product_catalog(self) -> dict[str, dict]:
        return {p["id"]: p for p in self.products["products"]}

    # ------------------------------------------------------------ Soporte ---
    def tickets_in(self, offsets: Iterable[int]) -> list[dict]:
        offs = set(offsets)
        return [t for t in self.tickets if t["day_offset"] in offs]


@lru_cache(maxsize=1)
def load_dataset() -> FixtureDataset:
    ga4 = _load(ROOT / "ga4" / "daily_metrics.json")
    shop = _load(ROOT / "shopify" / "daily_orders.json")
    return FixtureDataset(
        meta=_load(ROOT / "meta.json"),
        ga4_meta=ga4["meta"],
        ga4_rows=ga4["rows"],
        ga4_shares=_load(ROOT / "ga4" / "dimension_shares.json"),
        shopify_meta=shop["meta"],
        shopify_rows=shop["rows"],
        shopify_shares=_load(ROOT / "shopify" / "dimension_shares.json"),
        products=_load(ROOT / "shopify" / "products.json"),
        tickets=_load(ROOT / "support" / "tickets.json")["tickets"],
    )


def allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Reparte un entero entre valores de dimensión sin perder ni ganar unidades."""
    if not shares:
        return {}
    items = list(shares.items())
    raw = [total * s for _, s in items]
    base = [int(x) for x in raw]
    rest = total - sum(base)
    order = sorted(range(len(items)), key=lambda i: (-(raw[i] - base[i]), items[i][0]))
    for i in order[:max(rest, 0)]:
        base[i] += 1
    return {items[i][0]: base[i] for i in range(len(items))}


def allocate_money(total_eur: float, shares: dict[str, float]) -> dict[str, float]:
    cents = allocate(round(total_eur * 100), shares)
    return {k: v / 100 for k, v in cents.items()}
