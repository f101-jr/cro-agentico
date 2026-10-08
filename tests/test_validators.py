"""Validación post-hoc: el control más importante del sistema."""

from __future__ import annotations

from agent.guardrails import Budget
from agent.orchestrator import Orchestrator
from agent.providers.scripted import ScriptedProvider
from agent.validators import (
    SupportedFigures,
    apply_causal_rule,
    extract_figures,
    parse_number_candidates,
    validate_insight,
)
from semantic.schema import Insight
from tests.stubs import StubProvider, evidence_from, insight_payload, json_response, tool_response


def test_number_parsing_handles_spanish_and_english_notation():
    assert 51_400.0 in parse_number_candidates("51.400")
    assert 73_944.0 in parse_number_candidates("73.944,00")
    assert 1.8444 in parse_number_candidates("1,8444")
    assert 1_012.0 in parse_number_candidates("1.012")
    assert 12.15 in parse_number_candidates("12,15")


def test_window_tokens_and_ids_are_not_treated_as_figures():
    tokens = [t for t, _ in extract_figures("En last_14d (call_1_2, P-001) subió un 3%")]
    assert tokens == ["3"]


async def test_a_number_that_never_appeared_in_a_tool_is_detected(ctx):
    orchestrator = Orchestrator(ScriptedProvider(), ctx)
    result = await orchestrator.run("¿Cuántos pedidos hubo en los últimos 14 días?")
    insight = result.canonical_insights[0]
    inventado = insight.model_copy(update={
        "hypothesis": insight.hypothesis + " El margen bruto fue del 47,3%."
    })
    report = validate_insight(inventado, result.trace.tool_calls)
    assert report.ok is False
    assert "47,3" in report.orphan_figures
    assert "unsupported_figures" in report.codes()


async def test_real_answer_passes_the_numeric_validator(ctx):
    orchestrator = Orchestrator(ScriptedProvider(), ctx)
    result = await orchestrator.run("La conversión en móvil ha caído. ¿Por qué?")
    for insight in result.canonical_insights:
        assert validate_insight(insight, result.trace.tool_calls).ok


async def test_invented_figure_triggers_one_retry_and_then_passes(ctx):
    """Primer fallo -> reintento con las cifras huérfanas como feedback."""
    first = tool_response(("get_orders_summary", {"window": "last_14d", "dimension": "none"}))
    provider = StubProvider(responses=[first])
    orchestrator = Orchestrator(provider, ctx)

    # Primera pasada solo para tener una traza con observaciones reales.
    warmup = await Orchestrator(ScriptedProvider(), ctx).run("¿Cuántos pedidos hubo?")
    good_evidence = evidence_from(warmup, "orders", tool_call_id="call_0")

    bad = json_response(insight_payload("Los pedidos cayeron un 91,7% hasta 12.345 unidades.",
                                        [good_evidence]))
    good = json_response(insight_payload("Los pedidos de la ventana son los que devuelve la tool.",
                                         [good_evidence]))
    provider.responses = [first, bad, good]
    result = await orchestrator.run("¿Cuántos pedidos hubo?")
    assert result.validation_retries == 1
    assert result.degraded is False
    assert "12.345" not in result.insights[0].hypothesis


async def test_second_failure_degrades_with_unsupported_figures(ctx):
    warmup = await Orchestrator(ScriptedProvider(), ctx).run("¿Cuántos pedidos hubo?")
    good_evidence = evidence_from(warmup, "orders", tool_call_id="call_0")
    bad = json_response(insight_payload("Han caído un 91,7% hasta 12.345 pedidos por el envío.",
                                        [good_evidence], causal=False))
    provider = StubProvider(responses=[
        tool_response(("get_orders_summary", {"window": "last_14d", "dimension": "none"})),
        bad, bad,
    ])
    result = await Orchestrator(provider, ctx).run("¿Cuántos pedidos hubo?")
    insight = result.insights[0]
    assert insight.degraded is True
    assert insight.degraded_reason == "unsupported_figures"
    assert insight.causal_claim is False
    assert any("12.345" in a for a in insight.assumptions)


async def test_causal_claim_with_a_single_source_is_degraded(ctx):
    warmup = await Orchestrator(ScriptedProvider(), ctx).run("¿Cuántos pedidos hubo?")
    evidence = evidence_from(warmup, "orders", tool_call_id="call_0")
    provider = StubProvider(responses=[
        tool_response(("get_orders_summary", {"window": "last_14d", "dimension": "none"})),
        json_response(insight_payload("El cambio de envío ha causado la caída.", [evidence],
                                      causal=True, confidence="high")),
    ])
    result = await Orchestrator(provider, ctx).run("¿por qué?")
    insight = result.insights[0]
    assert insight.causal_claim is False
    assert insight.confidence == "low"
    assert any("una sola fuente" in a for a in insight.assumptions)


def test_apply_causal_rule_keeps_multi_source_claims():
    insight = Insight.model_validate({
        "id": "i1", "run_id": "r1", "hypothesis": "h", "causal_claim": True, "confidence": "high",
        "evidence": [
            {"tool_call_id": "c1", "metric": "orders", "value": 1, "window": "last_14d",
             "dimensions": {}, "source": "shopify", "quality": "exact"},
            {"tool_call_id": "c1", "metric": "sessions", "value": 2, "window": "last_14d",
             "dimensions": {}, "source": "ga4", "quality": "exact"},
        ],
    })
    same, changed = apply_causal_rule(insight)
    assert changed is False and same.causal_claim is True


async def test_evidence_with_unknown_tool_call_id_fails_validation(ctx):
    warmup = await Orchestrator(ScriptedProvider(), ctx).run("¿Cuántos pedidos hubo?")
    insight = warmup.canonical_insights[0]
    broken = insight.model_copy(update={
        "evidence": [insight.evidence[0].model_copy(update={"tool_call_id": "call_fantasma"})]
    })
    report = validate_insight(broken, warmup.trace.tool_calls)
    assert report.ok is False and "unknown_tool_call" in report.codes()


async def test_evidence_value_must_match_the_observation(ctx):
    warmup = await Orchestrator(ScriptedProvider(), ctx).run("¿Cuántos pedidos hubo?")
    insight = warmup.canonical_insights[0]
    broken = insight.model_copy(update={
        "evidence": [insight.evidence[0].model_copy(update={"value": 999_999.0})]
    })
    report = validate_insight(broken, warmup.trace.tool_calls)
    assert "evidence_value_mismatch" in report.codes()


async def test_guidance_text_counts_as_a_tool_result(ctx):
    """Si una tool dice 'n=12', citar 12 no es inventar."""
    orchestrator = Orchestrator(ScriptedProvider(), ctx)
    result = await orchestrator.run("¿Cuál es la conversión en tablet esta semana?")
    assert result.insights
    for insight in result.canonical_insights:
        assert validate_insight(insight, result.trace.tool_calls).ok


def test_supported_figures_accept_ratio_written_as_percentage():
    from datetime import date, datetime, timezone

    from semantic.schema import DataQuality, Observation, Window

    obs = Observation(metric="cr", value=0.121547, unit="ratio", window=Window.LAST_14D,
                      window_start=date(2026, 1, 1), window_end=date(2026, 1, 14), source="derived",
                      quality=DataQuality.DERIVED, fetched_at=datetime.now(timezone.utc))
    supported = SupportedFigures()
    supported.add_observation(obs)
    assert supported.supports([12.15])
    assert supported.supports([0.1215])
    assert not supported.supports([47.3])
