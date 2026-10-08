"""La persona solo cambia la síntesis, nunca las cifras."""

from __future__ import annotations

import pytest

from agent.orchestrator import Orchestrator, _sample_sizes
from agent.personas import PERSONAS, UnknownPersona, apply_persona, load_persona
from agent.providers.scripted import ScriptedProvider

QUESTION = "La conversión en móvil ha caído. ¿Por qué?"


@pytest.fixture
async def run(ctx):
    return await Orchestrator(ScriptedProvider(), ctx).run(QUESTION, persona="analyst")


def _figures(insights) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for insight in insights:
        for evidence in insight.evidence:
            key = (evidence.metric, tuple(sorted(evidence.dimensions.items())), evidence.window)
            out[key] = evidence.value
    return out


async def test_same_run_rendered_with_three_personas_has_identical_figures(run):
    sizes = _sample_sizes(run.trace)
    renders = {p: apply_persona(run.canonical_insights, p, sample_sizes=sizes) for p in PERSONAS}
    figures = {p: _figures(v) for p, v in renders.items()}
    common = set.intersection(*(set(f) for f in figures.values()))
    assert common, "las tres personas deberían compartir alguna métrica"
    for key in common:
        values = {figures[p][key] for p in PERSONAS}
        assert len(values) == 1, f"divergencia numérica en {key}: {values}"


async def test_ceo_never_returns_more_than_three_insights(ctx):
    orchestrator = Orchestrator(ScriptedProvider(), ctx)
    result = await orchestrator.run(QUESTION, persona="ceo")
    assert len(result.insights) <= 3
    assert load_persona("ceo").max_insights == 3


async def test_ceo_hides_assumptions_without_touching_evidence(run):
    ceo = apply_persona(run.canonical_insights, "ceo")
    assert all(i.assumptions == [] for i in ceo)
    assert all(i.evidence for i in ceo)


async def test_ecommerce_manager_always_returns_a_proposed_action(ctx):
    orchestrator = Orchestrator(ScriptedProvider(), ctx)
    for question in (QUESTION, "¿Cuántos pedidos hubo en los últimos 14 días?",
                     "¿Qué producto deja más margen?"):
        result = await orchestrator.run(question, persona="ecommerce_manager")
        assert result.insights
        assert all(i.proposed_action is not None for i in result.insights), question


async def test_analyst_keeps_full_evidence_and_sample_sizes(run):
    analyst = apply_persona(run.canonical_insights, "analyst", sample_sizes=_sample_sizes(run.trace))
    ceo = apply_persona(run.canonical_insights, "ceo")
    assert len(analyst[0].evidence) >= len(ceo[0].evidence)
    assert any("n=" in a for a in analyst[0].assumptions)


def test_unknown_persona_raises():
    with pytest.raises(UnknownPersona):
        load_persona("cfo")


def test_persona_prompts_declare_their_limits():
    assert load_persona("ecommerce_manager").proposed_action == "required"
    assert load_persona("analyst").max_insights is None
    assert load_persona("ceo").assumptions == "hidden"


async def test_summarized_evidence_keeps_one_entry_per_source(run):
    """Una causa resumida sigue enseñando las fuentes que la sostienen."""
    canonical = run.canonical_insights[0]
    assert canonical.causal_claim is True
    for persona in ("ceo", "ecommerce_manager"):
        projected = apply_persona(run.canonical_insights, persona)[0]
        assert {e.source for e in projected.evidence} == {e.source for e in canonical.evidence}
        assert len(projected.evidence) <= len(canonical.evidence)
