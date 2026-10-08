"""Generador determinista del dataset de referencia.

Escenario: una tienda sube el umbral de envío gratuito de 50 € a 75 € a mitad
de `last_14d` (el cambio está vivo en los offsets 1..7, es decir en `last_7d`).
Efecto: los carritos pequeños abandonan en checkout, la conversión cae, el AOV
sube y los ingresos quedan casi planos. Ninguna fuente aislada revela la causa.

Los fixtures se indexan por `day_offset` (1 = ayer), NO por fecha absoluta: así
el dataset es válido cualquier día que se ejecute la demo o los tests.

Uso:  uv run python -m fixtures.generate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Periodos: rangos de day_offset (inclusive). Ver connectors/windows.py
# --------------------------------------------------------------------------- #
PERIODS: dict[str, tuple[int, int]] = {
    "post7": (1, 7),  # umbral de envío ya en 75 €  -> last_7d
    "pre7": (8, 14),  # umbral todavía en 50 €      -> completa last_14d
    "p15_21": (15, 21),  # prev_7d
    "p22_28": (22, 28),  # completa prev_14d
    "p29_30": (29, 30),  # completa last_30d
    "p31_60": (31, 60),  # prev_30d
    "p61_64": (61, 64),  # colchón
    "sply": (365, 378),  # same_period_last_year
}

# Peso relativo de cada día dentro de un periodo, ciclado por (offset % 7).
# Suma 7.0, de modo que la media es 1.0 y el reparto es estable y determinista.
WEEKDAY_WEIGHTS = [1.05, 1.02, 0.97, 0.95, 1.00, 1.06, 0.95]

# --------------------------------------------------------------------------- #
# Objetivos por (device, periodo). Toda cifra del escenario sale de aquí.
# revenue en euros; los buckets suman exactamente orders y revenue.
# --------------------------------------------------------------------------- #
B = ("lt_50", "50_75", "gt_75")

TARGETS: dict[str, dict[str, dict[str, Any]]] = {
    "mobile": {
        "post7": dict(sessions=26000, carts=3538, checkouts=1256, orders=417,
                      buckets={"lt_50": (30, 1200.00), "50_75": (25, 1575.00), "gt_75": (362, 33304.00)}),
        "pre7": dict(sessions=25400, carts=3658, checkouts=1335, orders=531,
                     buckets={"lt_50": (156, 5772.00), "50_75": (107, 6580.50), "gt_75": (268, 25512.50)}),
        "p15_21": dict(sessions=24300, carts=3499, checkouts=1267, orders=509,
                       buckets={"lt_50": (160, 6080.00), "50_75": (133, 8246.00), "gt_75": (216, 21800.88)}),
        "p22_28": dict(sessions=23900, carts=3442, checkouts=1246, orders=503,
                       buckets={"lt_50": (158, 6004.00), "50_75": (131, 8122.00), "gt_75": (214, 21597.12)}),
        "p29_30": dict(sessions=6800, carts=979, checkouts=354, orders=143,
                       buckets={"lt_50": (45, 1710.00), "50_75": (37, 2294.00), "gt_75": (61, 6156.00)}),
        "p31_60": dict(sessions=102000, carts=14688, checkouts=5317, orders=2140,
                       buckets={"lt_50": (672, 25536.00), "50_75": (558, 34596.00), "gt_75": (910, 91819.00)}),
        "p61_64": dict(sessions=13600, carts=1958, checkouts=709, orders=286,
                       buckets={"lt_50": (90, 3420.00), "50_75": (74, 4588.00), "gt_75": (122, 12312.00)}),
        "sply": dict(sessions=41000, carts=5822, checkouts=2090, orders=848,
                     buckets={"lt_50": (280, 10360.00), "50_75": (220, 13420.00), "gt_75": (348, 33852.00)}),
    },
    "desktop": {
        "post7": dict(sessions=15900, carts=2544, checkouts=1145, orders=536,
                      buckets={"lt_50": (115, 4370.00), "50_75": (122, 7564.00), "gt_75": (299, 25330.00)}),
        "pre7": dict(sessions=15900, carts=2544, checkouts=1145, orders=539,
                     buckets={"lt_50": (125, 4750.00), "50_75": (128, 7936.00), "gt_75": (286, 24225.00)}),
        "p15_21": dict(sessions=15500, carts=2480, checkouts=1116, orders=531,
                       buckets={"lt_50": (150, 5700.00), "50_75": (140, 8680.00), "gt_75": (241, 22013.00)}),
        "p22_28": dict(sessions=15500, carts=2480, checkouts=1116, orders=529,
                       buckets={"lt_50": (150, 5700.00), "50_75": (140, 8680.00), "gt_75": (239, 21837.00)}),
        "p29_30": dict(sessions=4400, carts=704, checkouts=317, orders=150,
                       buckets={"lt_50": (42, 1596.00), "50_75": (40, 2480.00), "gt_75": (68, 6200.00)}),
        "p31_60": dict(sessions=66000, carts=10560, checkouts=4752, orders=2250,
                       buckets={"lt_50": (640, 24320.00), "50_75": (600, 37200.00), "gt_75": (1010, 92600.00)}),
        "p61_64": dict(sessions=8800, carts=1408, checkouts=634, orders=300,
                       buckets={"lt_50": (85, 3230.00), "50_75": (80, 4960.00), "gt_75": (135, 12360.00)}),
        "sply": dict(sessions=28000, carts=4480, checkouts=1970, orders=940,
                     buckets={"lt_50": (280, 10640.00), "50_75": (250, 15500.00), "gt_75": (410, 39000.00)}),
    },
    "tablet": {
        "post7": dict(sessions=620, carts=81, checkouts=27, orders=12,
                      buckets={"lt_50": (2, 76.00), "50_75": (2, 124.00), "gt_75": (8, 744.00)}),
        "pre7": dict(sessions=630, carts=82, checkouts=28, orders=15,
                     buckets={"lt_50": (4, 152.00), "50_75": (3, 186.00), "gt_75": (8, 772.00)}),
        "p15_21": dict(sessions=590, carts=77, checkouts=26, orders=15,
                       buckets={"lt_50": (5, 190.00), "50_75": (4, 248.00), "gt_75": (6, 606.00)}),
        "p22_28": dict(sessions=590, carts=77, checkouts=26, orders=14,
                       buckets={"lt_50": (4, 152.00), "50_75": (4, 248.00), "gt_75": (6, 606.00)}),
        "p29_30": dict(sessions=170, carts=22, checkouts=7, orders=4,
                       buckets={"lt_50": (1, 38.00), "50_75": (1, 62.00), "gt_75": (2, 180.00)}),
        "p31_60": dict(sessions=2550, carts=331, checkouts=112, orders=62,
                       buckets={"lt_50": (20, 760.00), "50_75": (18, 1116.00), "gt_75": (24, 2280.00)}),
        "p61_64": dict(sessions=340, carts=44, checkouts=15, orders=8,
                       buckets={"lt_50": (3, 114.00), "50_75": (2, 124.00), "gt_75": (3, 285.00)}),
        "sply": dict(sessions=1100, carts=140, checkouts=47, orders=24,
                     buckets={"lt_50": (9, 342.00), "50_75": (7, 434.00), "gt_75": (8, 760.00)}),
    },
}

USERS_PER_SESSION = 0.78

# --------------------------------------------------------------------------- #
# Reparto por dimensión. Las cuotas son estables entre periodos: el colapso del
# escenario es de DEVICE, no de canal; que sea transversal por canal es parte
# del hallazgo (obliga a cruzar fuentes para explicarlo).
# --------------------------------------------------------------------------- #
CHANNEL_SHARES = {
    "mobile": {"organic_search": 0.34, "paid_search": 0.26, "direct": 0.18, "email": 0.12, "social": 0.10},
    "desktop": {"organic_search": 0.40, "paid_search": 0.20, "direct": 0.24, "email": 0.10, "social": 0.06},
    "tablet": {"organic_search": 0.38, "paid_search": 0.18, "direct": 0.26, "email": 0.12, "social": 0.06},
}
# Índice de conversión relativo por canal (email convierte mejor, social peor).
CHANNEL_CR_INDEX = {"organic_search": 1.00, "paid_search": 0.90, "direct": 1.35, "email": 1.80, "social": 0.50}

DIMENSION_SHARES: dict[str, dict[str, dict[str, float]]] = {
    "source_medium": {
        "mobile": {"google / organic": 0.31, "google / cpc": 0.24, "(direct) / (none)": 0.18,
                   "newsletter / email": 0.12, "instagram / social": 0.08, "meta / cpc": 0.04, "bing / organic": 0.03},
        "desktop": {"google / organic": 0.36, "google / cpc": 0.18, "(direct) / (none)": 0.24,
                    "newsletter / email": 0.10, "instagram / social": 0.04, "meta / cpc": 0.03, "bing / organic": 0.05},
        "tablet": {"google / organic": 0.35, "google / cpc": 0.16, "(direct) / (none)": 0.26,
                   "newsletter / email": 0.12, "instagram / social": 0.05, "meta / cpc": 0.02, "bing / organic": 0.04},
    },
    "landing_page": {
        "mobile": {"/": 0.29, "/coleccion/novedades": 0.18, "/producto/bolso-tote": 0.14,
                   "/coleccion/rebajas": 0.13, "/blog/guia-de-tallas": 0.11,
                   "/coleccion/accesorios": 0.09, "/carrito": 0.06},
        "desktop": {"/": 0.32, "/coleccion/novedades": 0.17, "/producto/bolso-tote": 0.12,
                    "/coleccion/rebajas": 0.12, "/blog/guia-de-tallas": 0.10,
                    "/coleccion/accesorios": 0.11, "/carrito": 0.06},
        "tablet": {"/": 0.31, "/coleccion/novedades": 0.18, "/producto/bolso-tote": 0.13,
                   "/coleccion/rebajas": 0.12, "/blog/guia-de-tallas": 0.11,
                   "/coleccion/accesorios": 0.09, "/carrito": 0.06},
    },
    "browser": {
        "mobile": {"Chrome": 0.46, "Safari": 0.38, "Samsung Internet": 0.11, "Firefox": 0.03, "Edge": 0.02},
        "desktop": {"Chrome": 0.55, "Safari": 0.18, "Edge": 0.15, "Firefox": 0.10, "Samsung Internet": 0.02},
        "tablet": {"Safari": 0.58, "Chrome": 0.33, "Firefox": 0.05, "Edge": 0.03, "Samsung Internet": 0.01},
    },
    "os": {
        "mobile": {"Android": 0.55, "iOS": 0.45},
        "desktop": {"Windows": 0.62, "macOS": 0.31, "Linux": 0.07},
        "tablet": {"iOS": 0.68, "Android": 0.32},
    },
    "country": {
        "mobile": {"ES": 0.74, "FR": 0.08, "PT": 0.07, "IT": 0.05, "DE": 0.04, "MX": 0.02},
        "desktop": {"ES": 0.71, "FR": 0.09, "PT": 0.06, "IT": 0.06, "DE": 0.05, "MX": 0.03},
        "tablet": {"ES": 0.76, "FR": 0.07, "PT": 0.06, "IT": 0.05, "DE": 0.04, "MX": 0.02},
    },
}

# Índice de conversión relativo por valor de dimensión: permite que la conversión
# varíe entre segmentos sin romper los totales (los pedidos se reparten con estos
# pesos y luego se cuadran al total exacto).
DIMENSION_CR_INDEX: dict[str, dict[str, float]] = {
    "channel": CHANNEL_CR_INDEX,
    "source_medium": {"google / organic": 1.00, "google / cpc": 0.90, "(direct) / (none)": 1.35,
                      "newsletter / email": 1.80, "instagram / social": 0.50, "meta / cpc": 0.55,
                      "bing / organic": 0.95},
    "landing_page": {"/": 0.90, "/coleccion/novedades": 1.10, "/producto/bolso-tote": 1.60,
                     "/coleccion/rebajas": 1.20, "/blog/guia-de-tallas": 0.35,
                     "/coleccion/accesorios": 1.00, "/carrito": 2.40},
    "browser": {"Chrome": 1.00, "Safari": 0.95, "Samsung Internet": 0.90, "Firefox": 1.00, "Edge": 1.05},
    "os": {"Android": 0.95, "iOS": 1.05, "Windows": 1.00, "macOS": 1.05, "Linux": 0.90},
    "country": {"ES": 1.08, "FR": 0.70, "PT": 0.80, "IT": 0.65, "DE": 0.60, "MX": 0.45},
}

PRODUCTS = [
    # id, título, tipo, precio €, cuota de unidades pre-cambio, cuota post-cambio
    ("P-001", "Bolso tote de lona", "bags", 89.00, 0.12, 0.14),
    ("P-002", "Cinturón de piel", "accessories", 45.00, 0.10, 0.09),
    ("P-003", "Zapatillas urbanas", "footwear", 79.00, 0.11, 0.12),
    ("P-004", "Camiseta algodón orgánico", "apparel", 29.00, 0.16, 0.12),
    ("P-005", "Sudadera con capucha", "apparel", 59.00, 0.12, 0.14),
    ("P-006", "Mochila 20L", "bags", 75.00, 0.09, 0.11),
    ("P-007", "Gorra con logo", "accessories", 25.00, 0.08, 0.06),
    ("P-008", "Pack 3 calcetines", "accessories", 19.00, 0.11, 0.08),
    ("P-009", "Chaqueta ligera", "apparel", 119.00, 0.05, 0.08),
    ("P-010", "Bufanda de lana", "accessories", 39.00, 0.06, 0.06),
]
ITEMS_PER_ORDER = 1.6

# --------------------------------------------------------------------------- #
# Soporte: asuntos reales; el conector clasifica por palabras clave EN CÓDIGO.
# --------------------------------------------------------------------------- #
SUBJECTS: dict[str, list[str]] = {
    "shipping": [
        "¿Por qué me cobran envío si antes era gratis?",
        "El envío gratuito ya no me aparece en el carrito",
        "Gastos de envío demasiado altos para un pedido pequeño",
        "¿A partir de qué importe el envío es gratuito?",
        "Me han cobrado portes sin avisar",
    ],
    "payment": [
        "El pago con tarjeta me da error",
        "No puedo pagar con Bizum",
        "Cobro duplicado en mi tarjeta",
    ],
    "product": [
        "¿Qué talla es la sudadera con capucha?",
        "El artículo llegó con un defecto de costura",
        "¿Hay stock del bolso en color negro?",
    ],
    "returns": [
        "Quiero tramitar una devolución",
        "¿Cuánto tarda el reembolso?",
        "No me llega la etiqueta para la devolución",
    ],
    "account": [
        "No puedo acceder a mi cuenta",
        "Restablecer la contraseña de mi cuenta",
        "Cambiar los datos de mi cuenta",
    ],
}

# Tickets por periodo y tema. shipping x3,2 en last_14d frente a prev_14d.
TICKETS: dict[str, dict[str, int]] = {
    "post7": {"shipping": 83, "payment": 12, "product": 17, "returns": 14, "account": 5},
    "pre7": {"shipping": 45, "payment": 12, "product": 16, "returns": 11, "account": 6},
    "p15_21": {"shipping": 21, "payment": 11, "product": 18, "returns": 9, "account": 6},
    "p22_28": {"shipping": 19, "payment": 11, "product": 17, "returns": 9, "account": 6},
    "p29_30": {"shipping": 6, "payment": 3, "product": 5, "returns": 3, "account": 2},
    "p31_60": {"shipping": 84, "payment": 47, "product": 74, "returns": 39, "account": 26},
    "p61_64": {"shipping": 11, "payment": 6, "product": 10, "returns": 5, "account": 3},
    "sply": {"shipping": 35, "payment": 19, "product": 31, "returns": 16, "account": 11},
}


# --------------------------------------------------------------------------- #
# Utilidades de reparto (método del mayor resto: la suma es EXACTA)
# --------------------------------------------------------------------------- #
def distribute(total: int, weights: list[float]) -> list[int]:
    """Reparte `total` (entero) entre len(weights) celdas respetando los pesos."""
    if total == 0 or not weights:
        return [0] * len(weights)
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    base = [int(x) for x in raw]
    rest = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base


def distribute_cents(total_eur: float, weights: list[float]) -> list[float]:
    cents = distribute(round(total_eur * 100), weights)
    return [c / 100 for c in cents]


def period_offsets(period: str) -> list[int]:
    lo, hi = PERIODS[period]
    return list(range(lo, hi + 1))


def day_weights(offsets: list[int]) -> list[float]:
    return [WEEKDAY_WEIGHTS[o % 7] for o in offsets]


# --------------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------------- #
def build_ga4_rows() -> list[dict]:
    rows: list[dict] = []
    for device, periods in TARGETS.items():
        for period, t in periods.items():
            offs = period_offsets(period)
            w = day_weights(offs)
            sessions = distribute(t["sessions"], w)
            carts = distribute(t["carts"], w)
            checkouts = distribute(t["checkouts"], w)
            purchases = distribute(t["orders"], w)
            users = [round(s * USERS_PER_SESSION) for s in sessions]
            for i, off in enumerate(offs):
                rows.append({
                    "day_offset": off,
                    "device": device,
                    "period": period,
                    "sessions": sessions[i],
                    "users": users[i],
                    "views": sessions[i],
                    "carts": carts[i],
                    "checkouts": checkouts[i],
                    "purchases": purchases[i],
                })
    rows.sort(key=lambda r: (r["day_offset"], r["device"]))
    return rows


def build_shopify_rows() -> list[dict]:
    rows: list[dict] = []
    for device, periods in TARGETS.items():
        for period, t in periods.items():
            offs = period_offsets(period)
            w = day_weights(offs)
            per_bucket_orders = {b: distribute(t["buckets"][b][0], w) for b in B}
            # El importe sigue al pedido: un día sin pedidos en un bucket no puede
            # tener ingresos en ese bucket (si no, el revenue se evapora al agregar).
            per_bucket_rev = {
                b: distribute_cents(t["buckets"][b][1],
                                    [float(x) for x in per_bucket_orders[b]] if any(per_bucket_orders[b]) else w)
                for b in B
            }
            for i, off in enumerate(offs):
                buckets = {
                    b: {"orders": per_bucket_orders[b][i], "revenue_eur": round(per_bucket_rev[b][i], 2)}
                    for b in B
                }
                rows.append({
                    "day_offset": off,
                    "device": device,
                    "period": period,
                    "orders": sum(buckets[b]["orders"] for b in B),
                    "revenue_eur": round(sum(buckets[b]["revenue_eur"] for b in B), 2),
                    "buckets": buckets,
                })
    rows.sort(key=lambda r: (r["day_offset"], r["device"]))
    return rows


def channel_order_shares() -> dict[str, dict[str, float]]:
    """Cuota de PEDIDOS por canal = cuota de sesiones x índice de conversión."""
    out: dict[str, dict[str, float]] = {}
    for device, shares in CHANNEL_SHARES.items():
        weighted = {c: s * CHANNEL_CR_INDEX[c] for c, s in shares.items()}
        total = sum(weighted.values())
        out[device] = {c: round(v / total, 6) for c, v in weighted.items()}
    return out


def build_tickets() -> list[dict]:
    tickets: list[dict] = []
    seq = 0
    for period, topics in TICKETS.items():
        offs = period_offsets(period)
        w = day_weights(offs)
        for topic, total in topics.items():
            per_day = distribute(total, w)
            subjects = SUBJECTS[topic]
            for i, off in enumerate(offs):
                for k in range(per_day[i]):
                    seq += 1
                    tickets.append({
                        "id": f"TCK-{seq:05d}",
                        "day_offset": off,
                        "subject": subjects[(off + k + seq) % len(subjects)],
                        "channel": ("email", "chat", "form")[(seq + off) % 3],
                        "intended_topic": topic,  # solo para fixtures/validate.py
                    })
    tickets.sort(key=lambda t: (t["day_offset"], t["id"]))
    return tickets


SCENARIO = (
    "El umbral de envío gratuito pasa de 50 € a 75 € y el cambio está vivo en "
    "los últimos 7 días (offsets 1..7). Los carritos pequeños abandonan en "
    "checkout: la conversión móvil cae, el AOV sube, los ingresos quedan casi "
    "planos y los tickets de envío se multiplican. Escritorio actúa de control."
)


def main() -> None:
    meta = {
        "scenario": SCENARIO,
        "indexing": "day_offset: 1 = ayer. Sin fechas absolutas: el dataset vale cualquier día.",
        "change_live_from_offset": 7,
        "periods": {k: list(v) for k, v in PERIODS.items()},
        "generator": "fixtures/generate.py",
    }
    (ROOT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ga4 = {
        "meta": {"property_timezone": "Europe/Madrid", "property_id": "properties/000000000",
                 "sampled": False, "consolidation_lag_days": 1,
                 "note": "Fixtures consolidados por construcción: last_complete_date = ayer."},
        "rows": build_ga4_rows(),
    }
    _write(ROOT / "ga4" / "daily_metrics.json", ga4)
    _write(ROOT / "ga4" / "dimension_shares.json", {
        "channel": CHANNEL_SHARES,
        "channel_cr_index": CHANNEL_CR_INDEX,
        "cr_index": DIMENSION_CR_INDEX,
        **DIMENSION_SHARES,
    })

    _write(ROOT / "shopify" / "daily_orders.json", {
        "meta": {"currency": "EUR", "filter": "financial_status != voided, cancelled_at IS NULL, "
                                              "revenue neto de devoluciones",
                 "buckets": {"lt_50": "< 50 €", "50_75": "50-75 €", "gt_75": "> 75 €"}},
        "rows": build_shopify_rows(),
    })
    _write(ROOT / "shopify" / "dimension_shares.json", {
        "channel_order_shares": channel_order_shares(),
    })
    _write(ROOT / "shopify" / "products.json", {
        "items_per_order": ITEMS_PER_ORDER,
        "products": [
            {"id": p[0], "title": p[1], "product_type": p[2], "price_eur": p[3],
             "unit_share_pre": p[4], "unit_share_post": p[5]}
            for p in PRODUCTS
        ],
    })

    _write(ROOT / "support" / "tickets.json", {
        "meta": {"note": "El tema NO viene etiquetado por la fuente: connectors/support.py lo "
                         "clasifica por palabras clave en código. `intended_topic` solo lo usa "
                         "fixtures/validate.py para comprobar el clasificador."},
        "tickets": build_tickets(),
    })
    print("fixtures generados en", ROOT)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
