"""Validador de invariantes del dataset.

Si un fixture rompe una invariante, `pytest` falla. No se generan fixtures a
mano sin pasar por este validador.

Uso:  uv run python -m fixtures.validate
"""

from __future__ import annotations

import sys

from connectors.windows import day_offsets
from fixtures.dataset import BUCKETS, DEVICES, load_dataset
from semantic.schema import Window

EUR_TOL = 0.01
RATIO_TOL = 0.0001

# Cifras normativas del escenario (móvil). Si esto deja de cuadrar, el dataset
# ha dejado de contar el escenario.
SCENARIO_MOBILE = {
    Window.PREV_14D: dict(sessions=48200, orders=1012, revenue=71850.00, aov=71.00, cr=0.0210,
                          buckets={"lt_50": 318, "50_75": 264, "gt_75": 430},
                          funnel=(0.144, 0.362, 0.403)),
    Window.LAST_14D: dict(sessions=51400, orders=948, revenue=73944.00, aov=78.00, cr=0.0184,
                          buckets={"lt_50": 186, "50_75": 132, "gt_75": 630},
                          funnel=(0.140, 0.360, 0.366)),
}
SCENARIO_DESKTOP_CR = {Window.PREV_14D: 0.0342, Window.LAST_14D: 0.0338}
SCENARIO_SHIPPING_RATIO = 3.2


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def validate() -> list[str]:
    d = load_dataset()
    problems: list[str] = []
    windows = [w for w in Window]

    # --- Invariantes genéricas, toda ventana y toda dimensión -------------- #
    for w in windows:
        offs = day_offsets(w)
        for device in ("all", *DEVICES):
            g = d.ga4_totals(offs, device)
            s = d.shopify_totals(offs, device)

            if s["orders"] != sum(s["buckets"][b]["orders"] for b in BUCKETS):
                problems.append(f"{w}/{device}: orders != suma de orders por aov_bucket")
            if not _close(s["revenue_eur"], sum(s["buckets"][b]["revenue_eur"] for b in BUCKETS), EUR_TOL):
                problems.append(f"{w}/{device}: revenue != suma de revenue por aov_bucket")

            if s["orders"]:
                aov = s["revenue_eur"] / s["orders"]
                if not _close(aov, round(aov, 2), EUR_TOL):
                    problems.append(f"{w}/{device}: revenue/orders != aov")
            for b in BUCKETS:
                bo, br = s["buckets"][b]["orders"], s["buckets"][b]["revenue_eur"]
                if bo and not _close(br / bo, round(br / bo, 2), EUR_TOL):
                    problems.append(f"{w}/{device}/{b}: revenue/orders != aov")
                bounds = {"lt_50": (0, 50), "50_75": (50, 75), "gt_75": (75, 10_000)}[b]
                if bo and not (bounds[0] <= br / bo <= bounds[1]):
                    problems.append(f"{w}/{device}/{b}: ticket medio {br / bo:.2f} € fuera del bucket")

            if g["sessions"]:
                cr = s["orders"] / g["sessions"]
                if not _close(cr, s["orders"] / g["sessions"], RATIO_TOL):
                    problems.append(f"{w}/{device}: conversion_rate != orders/sessions")

            steps = [g["views"], g["carts"], g["checkouts"], g["purchases"]]
            if any(steps[i] < steps[i + 1] for i in range(3)):
                problems.append(f"{w}/{device}: escalones del funnel no monótonos {steps}")

            if g["purchases"] != s["orders"]:
                problems.append(f"{w}/{device}: purchases GA4 ({g['purchases']}) != orders Shopify ({s['orders']})")

    # --- Coherencia diaria de los buckets ---------------------------------- #
    for row in d.shopify_rows:
        for b in BUCKETS:
            bo, br = row["buckets"][b]["orders"], row["buckets"][b]["revenue_eur"]
            if (bo == 0) != (br == 0):
                problems.append(f"día {row['day_offset']}/{row['device']}/{b}: orders={bo} pero "
                                f"revenue={br}: el importe no sigue al pedido")
            bounds = {"lt_50": (0, 50), "50_75": (50, 75), "gt_75": (75, 10_000)}[b]
            if bo and not (bounds[0] <= br / bo <= bounds[1]):
                problems.append(f"día {row['day_offset']}/{row['device']}/{b}: ticket medio "
                                f"{br / bo:.2f} € fuera del bucket")

    # --- Monotonía diaria -------------------------------------------------- #
    for row in d.ga4_rows:
        steps = [row["views"], row["carts"], row["checkouts"], row["purchases"]]
        if any(steps[i] < steps[i + 1] for i in range(3)):
            problems.append(f"día {row['day_offset']}/{row['device']}: funnel no monótono {steps}")

    # --- Cifras normativas del escenario ----------------------------------- #
    for w, exp in SCENARIO_MOBILE.items():
        offs = day_offsets(w)
        g = d.ga4_totals(offs, "mobile")
        s = d.shopify_totals(offs, "mobile")
        if g["sessions"] != exp["sessions"]:
            problems.append(f"{w}: sesiones móvil {g['sessions']} != {exp['sessions']}")
        if s["orders"] != exp["orders"]:
            problems.append(f"{w}: pedidos móvil {s['orders']} != {exp['orders']}")
        if not _close(s["revenue_eur"], exp["revenue"], EUR_TOL):
            problems.append(f"{w}: ingresos móvil {s['revenue_eur']} != {exp['revenue']}")
        if not _close(round(s["revenue_eur"] / s["orders"], 2), exp["aov"], EUR_TOL):
            problems.append(f"{w}: AOV móvil {s['revenue_eur'] / s['orders']:.2f} != {exp['aov']}")
        if not _close(round(s["orders"] / g["sessions"], 4), exp["cr"], 0.0001):
            problems.append(f"{w}: conversión móvil {s['orders'] / g['sessions']:.4f} != {exp['cr']}")
        for b, n in exp["buckets"].items():
            if s["buckets"][b]["orders"] != n:
                problems.append(f"{w}/{b}: pedidos {s['buckets'][b]['orders']} != {n}")
        rates = (g["carts"] / g["views"], g["checkouts"] / g["carts"], g["purchases"] / g["checkouts"])
        for got, want, name in zip(rates, exp["funnel"], ("view->cart", "cart->checkout", "checkout->purchase")):
            if not _close(round(got, 3), want, 0.001):
                problems.append(f"{w}/{name}: tasa {got:.3f} != {want}")

    for w, want in SCENARIO_DESKTOP_CR.items():
        offs = day_offsets(w)
        cr = d.shopify_totals(offs, "desktop")["orders"] / d.ga4_totals(offs, "desktop")["sessions"]
        if not _close(round(cr, 4), want, 0.0001):
            problems.append(f"{w}: conversión escritorio {cr:.4f} != {want}")

    # --- Soporte: tema clasificado EN CÓDIGO ------------------------------- #
    from connectors.support import classify_topic  # import tardío: evita ciclo

    for t in d.tickets:
        got = classify_topic(t["subject"])
        if got != t["intended_topic"]:
            problems.append(f"ticket {t['id']}: clasificado '{got}', esperado '{t['intended_topic']}' "
                            f"({t['subject']!r})")
            break  # un ejemplo basta; el resto sería ruido

    last = d.tickets_in(day_offsets(Window.LAST_14D))
    prev = d.tickets_in(day_offsets(Window.PREV_14D))
    n_last = sum(1 for t in last if classify_topic(t["subject"]) == "shipping")
    n_prev = sum(1 for t in prev if classify_topic(t["subject"]) == "shipping")
    ratio = n_last / n_prev if n_prev else 0
    if not _close(ratio, SCENARIO_SHIPPING_RATIO, 0.15):
        problems.append(f"tickets shipping: x{ratio:.2f} (last {n_last} / prev {n_prev}) != x{SCENARIO_SHIPPING_RATIO}")

    return problems


def main() -> int:
    problems = validate()
    if problems:
        print(f"FIXTURES INVÁLIDOS ({len(problems)} problemas):")
        for p in problems:
            print("  -", p)
        return 1
    print("fixtures OK: todas las invariantes del dataset se cumplen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
