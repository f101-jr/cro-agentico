"""Fuente de verdad por métrica.

Una métrica tiene UNA fuente. Cuando dos fuentes pueden responder a lo mismo,
esta tabla decide cuál manda; cuando una métrica se compone, `DERIVED_METRICS`
declara la fórmula y la nota que **debe** viajar en toda `Observation`.
"""

from __future__ import annotations

from semantic.schema import DataQuality

SOURCE_OF_TRUTH: dict[str, str] = {
    "sessions": "ga4",
    "users": "ga4",
    "funnel_steps": "ga4",
    "traffic_by_dim": "ga4",
    "orders": "shopify",
    "revenue": "shopify",
    "aov": "shopify",
    "refunds": "shopify",
    "support_tickets": "support",
    "conversion_rate": "derived",
}

DERIVED_METRICS: dict[str, dict] = {
    "conversion_rate": {
        "formula": "orders[shopify] / sessions[ga4]",
        "note": ("Orders de Shopify (excl. cancelados) sobre sesiones de GA4. "
                 "No coincide con la conversión que reporta GA4, que subcuenta "
                 "entre un 3% y un 8% por bloqueo de tracking."),
        "quality": DataQuality.DERIVED,
        "components": {"orders": "shopify", "sessions": "ga4"},
    },
}

# Métricas que el sistema NO puede responder con las fuentes conectadas. Están
# aquí para que la capa semántica pueda decir "no tengo ese dato" con precisión
# en lugar de devolver un vacío que el modelo rellenaría.
UNSUPPORTED_METRICS: dict[str, str] = {
    "margin": "coste de producto (COGS): ninguna fuente conectada lo expone",
    "gross_margin": "coste de producto (COGS): ninguna fuente conectada lo expone",
    "cogs": "coste de producto: ninguna fuente conectada lo expone",
    "cac": "gasto de medios: no hay conector de plataformas de publicidad",
    "roas": "gasto de medios: no hay conector de plataformas de publicidad",
    "ad_spend": "gasto de medios: no hay conector de plataformas de publicidad",
    "ltv": "histórico de cliente: se descarta en el conector por política de PII",
    "stock": "inventario: no hay conector de inventario",
    "nps": "encuestas: no hay conector de satisfacción",
}

COMPARABLE_METRICS = ("sessions", "users", "orders", "revenue", "aov", "conversion_rate",
                      "support_tickets", "refunds")


def source_for(metric: str) -> str | None:
    return SOURCE_OF_TRUTH.get(metric)


def is_derived(metric: str) -> bool:
    return metric in DERIVED_METRICS


def derived_note(metric: str) -> str | None:
    spec = DERIVED_METRICS.get(metric)
    return spec["note"] if spec else None


def unsupported_reason(metric: str) -> str | None:
    return UNSUPPORTED_METRICS.get(metric)


__all__ = [
    "COMPARABLE_METRICS",
    "DERIVED_METRICS",
    "SOURCE_OF_TRUTH",
    "UNSUPPORTED_METRICS",
    "derived_note",
    "is_derived",
    "source_for",
    "unsupported_reason",
]
