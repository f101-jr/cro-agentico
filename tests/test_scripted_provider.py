"""El analista determinista: cifras coherentes entre sí, no solo trazables.

El validador numérico comprueba que cada cifra existe en una observación, pero
no que sean la misma cifra comparable: un total frente a una fila de un desglose
pasarían el control y contarían una mentira. Eso se controla aquí.
"""

from __future__ import annotations

from agent.orchestrator import Orchestrator
from agent.providers.scripted import Index, ObservationRef, ScriptedProvider, classify
from api.main import DASHBOARD_QUESTION
from semantic.schema import Device, Window


def _ref(metric: str, value: float, window: str, dimensions: dict) -> ObservationRef:
    return ObservationRef("call_0", "t", {"metric": metric, "value": value, "window": window,
                                          "dimensions": dimensions, "source": "shopify",
                                          "quality": "exact", "unit": "currency_eur"})


def test_total_lookup_ignores_breakdown_rows():
    index = Index(refs=[
        _ref("aov", 95.94, "prev_14d", {"device": "all", "aov_bucket": ">75€"}),
        _ref("aov", 69.73, "prev_14d", {"device": "all"}),
    ])
    assert index.total("aov", window="prev_14d", device="all").value == 69.73
    # find() casa por subconjunto y por eso NO sirve para un total
    assert index.find("aov", window="prev_14d", device="all").value == 95.94
    assert index.total("aov", window="last_14d", device="all") is None


async def test_causal_narrative_cites_total_aov_not_a_breakdown_row(ctx):
    """Con device=all, el AOV de la ventana previa no puede ser el de un bucket."""
    result = await Orchestrator(ScriptedProvider(), ctx).run(DASHBOARD_QUESTION)
    insight = result.canonical_insights[0]
    aov = [e for e in insight.evidence if e.metric == "aov"]
    assert len(aov) == 2, "el relato compara el AOV de dos ventanas"
    assert all(set(e.dimensions) <= {"device"} for e in aov), \
        "se ha citado una fila de desglose como si fuera el total"
    assert {e.window for e in aov} == {Window.LAST_14D, Window.PREV_14D}


async def test_narrated_aov_delta_matches_the_two_cited_values(ctx):
    result = await Orchestrator(ScriptedProvider(), ctx).run(DASHBOARD_QUESTION)
    insight = result.canonical_insights[0]
    aov = {e.window: e.value for e in insight.evidence if e.metric == "aov"}
    delta = next(e.value for e in insight.evidence if e.metric == "aov_delta_rel")
    expected = (aov[Window.LAST_14D] - aov[Window.PREV_14D]) / aov[Window.PREV_14D]
    assert round(delta, 4) == round(expected, 4)


def test_classification_picks_device_window_and_dimension():
    plan = classify("¿Por qué ha caído la conversión en móvil esta semana?", Window.LAST_14D)
    assert (plan.kind, plan.device, plan.window) == ("causal_conversion", Device.MOBILE, Window.LAST_7D)
    assert plan.compare_window is Window.PREV_7D

    plan = classify("¿Qué producto deja más margen?", Window.LAST_14D)
    assert plan.kind == "unsupported" and "COGS" in plan.missing

    plan = classify("¿Cómo se reparten los pedidos por importe de cesta?", Window.LAST_14D)
    assert plan.dimension == "aov_bucket"


def test_year_over_year_is_read_the_right_way_round():
    plan = classify("¿Cómo va la conversión frente al mismo periodo del año pasado?", Window.LAST_14D)
    assert plan.window is Window.LAST_14D
    assert plan.compare_window is Window.SAME_PERIOD_LAST_YEAR
