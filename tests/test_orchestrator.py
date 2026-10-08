"""Orquestador: bucle, paralelismo, presupuesto y traza."""

from __future__ import annotations

import asyncio
import time

import pytest

from agent.guardrails import Budget
from agent.orchestrator import Orchestrator, load_system_prompt
from agent.providers.scripted import ScriptedProvider
from semantic.schema import Window
from semantic.tools import ToolResult
from storage.traces import InMemoryTraceStore
from tests.stubs import StubProvider, json_response, tool_response


def test_system_prompt_contains_the_hard_rules():
    version, body = load_system_prompt()
    assert version != "0"
    for rule in [
        "No calcules cifras. Toda cifra procede de un resultado de tool.",
        "Si una cifra no está en un resultado de tool, no puede aparecer en tu respuesta.",
        "Antes de afirmar una causa, consulta al menos dos fuentes distintas.",
        "marca `causal_claim: false`",
        "Si los datos no bastan, dilo y enumera qué falta. No rellenes.",
        "no promedies: recógelo en `contradictions`",
    ]:
        assert rule in body, rule
    for section in ("# Rol", "# Reglas duras", "# Protocolo", "# Salida", "# Alcance"):
        assert section in body


async def test_prompt_version_is_recorded_in_every_trace(ctx):
    orchestrator = Orchestrator(ScriptedProvider(), ctx, store=InMemoryTraceStore())
    result = await orchestrator.run("¿Cuántos pedidos hubo en los últimos 14 días?")
    assert result.trace.prompt_version == load_system_prompt()[0]
    assert result.trace.model == "scripted-offline"


async def test_independent_tool_calls_run_in_parallel(ctx, monkeypatch):
    """Dos tools lentas en la misma iteración no pueden tardar el doble."""
    inflight = 0
    peak = 0

    async def slow_execute(context, name, arguments, **kwargs):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        return ToolResult(tool=name, ok=True)

    monkeypatch.setattr("agent.orchestrator.execute_tool", slow_execute)
    provider = StubProvider(responses=[
        tool_response(("get_orders_summary", {"window": "last_14d"}),
                      ("get_support_signal", {"window": "last_14d"}),
                      ("get_funnel_metrics", {"window": "last_14d"})),
        json_response({"insights": [{
            "hypothesis": "sin cifras", "causal_claim": False, "confidence": "low",
            "evidence": [{"tool_call_id": "call_0", "metric": "orders", "value": 0,
                          "window": "last_14d", "dimensions": {}, "source": "shopify",
                          "quality": "exact"}],
        }]}),
    ])
    orchestrator = Orchestrator(provider, ctx)
    started = time.perf_counter()
    await orchestrator.run("pregunta")
    elapsed = time.perf_counter() - started
    assert peak == 3, "las tool_calls de una misma iteración deben ejecutarse a la vez"
    assert elapsed < 0.12, f"parecen secuenciales: {elapsed:.3f}s"


async def test_tool_calls_are_recorded_in_the_trace(ctx):
    orchestrator = Orchestrator(ScriptedProvider(), ctx)
    result = await orchestrator.run("La conversión en móvil ha caído. ¿Por qué?")
    assert len(result.trace.tool_calls) == 10
    assert all(call.latency_ms >= 0 for call in result.trace.tool_calls)
    assert {call.tool for call in result.trace.tool_calls} >= {
        "get_conversion_rate", "compare_windows", "get_funnel_metrics", "get_support_signal"}


async def test_exceeding_max_iterations_degrades_instead_of_failing(ctx):
    """Un proveedor que nunca cierra el análisis no rompe el run."""
    provider = StubProvider(responses=[tool_response(("get_orders_summary", {"window": "last_14d"}))])
    orchestrator = Orchestrator(provider, ctx, budget_factory=lambda: Budget(max_iterations=3))
    result = await orchestrator.run("dame una vuelta infinita")
    assert result.stop_reason == "max_iterations"
    assert result.degraded is True
    assert result.insights and result.insights[0].degraded_reason == "max_iterations"
    assert result.insights[0].causal_claim is False


async def test_max_tool_calls_is_enforced(ctx):
    provider = StubProvider(responses=[
        tool_response(*[("get_orders_summary", {"window": "last_14d"})] * 5),
    ])
    orchestrator = Orchestrator(provider, ctx, budget_factory=lambda: Budget(max_tool_calls=6))
    result = await orchestrator.run("muchas llamadas")
    assert len(result.trace.tool_calls) <= 6
    assert result.degraded is True


async def test_tool_timeout_becomes_guidance_not_crash(ctx, monkeypatch):
    async def hangs(context, name, arguments, **kwargs):
        await asyncio.sleep(1)
        return ToolResult(tool=name)

    monkeypatch.setattr("agent.orchestrator.execute_tool", hangs)
    provider = StubProvider(responses=[
        tool_response(("get_orders_summary", {"window": "last_14d"})),
        json_response({"insights": [{
            "hypothesis": "sin datos", "causal_claim": False, "confidence": "low",
            "evidence": [{"tool_call_id": "call_0", "metric": "orders", "value": 0,
                          "window": "last_14d", "dimensions": {}, "source": "shopify",
                          "quality": "exact"}]}]}),
    ])
    orchestrator = Orchestrator(provider, ctx, budget_factory=lambda: Budget(tool_timeout_s=0.05))
    result = await orchestrator.run("pregunta")
    assert result.trace.tool_calls[0].ok is False
    assert result.trace.tool_calls[0].error == "tool_timeout"


async def test_run_is_persisted_with_its_insights(ctx):
    store = InMemoryTraceStore()
    orchestrator = Orchestrator(ScriptedProvider(), ctx, store=store)
    result = await orchestrator.run("¿Cuántos pedidos hubo?", window=Window.LAST_7D)
    assert await store.get_trace(result.run_id) is not None
    assert (await store.get_insights(result.run_id))[0].id == result.insights[0].id


async def test_unknown_persona_is_rejected_before_spending_a_token(ctx):
    from agent.personas import UnknownPersona

    provider = StubProvider(responses=[json_response({"insights": []})])
    orchestrator = Orchestrator(provider, ctx)
    with pytest.raises(UnknownPersona):
        await orchestrator.run("pregunta", persona="cfo")
    assert provider.calls == 0
