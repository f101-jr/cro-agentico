"""El set dorado corre contra fixtures y sin red."""

from __future__ import annotations

import pytest

from config import Settings
from container import build_container
from evals.run import FAMILIES, load_cases, run_case, summarize


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_golden_has_at_least_25_cases_in_three_families(cases):
    assert len(cases) >= 25
    families = {c["family"] for c in cases}
    assert families == set(FAMILIES)
    assert len({c["id"] for c in cases}) == len(cases), "hay ids repetidos"


def test_refusal_family_cannot_be_emptied(cases):
    refusals = [c for c in cases if c["family"] == "refusal"]
    assert len(refusals) >= 3


def test_every_case_declares_at_least_one_expectation(cases):
    expectation_keys = {
        "must_call_tools", "must_cite_sources", "expect_contains", "forbid_contains",
        "expect_causal_claim", "expect_figure", "max_tool_calls", "min_evidence",
        "expect_degraded_or_assumption", "expect_degraded", "forbid_figures_not_in_tools",
        "max_insights",
    }
    for case in cases:
        assert expectation_keys & set(case), f"{case['id']} no comprueba nada"


@pytest.fixture
def eval_container():
    return build_container(Settings(mode="fixtures", llm_provider="scripted", trace_db=":memory:"))


async def test_the_three_reference_cases_pass(cases, eval_container):
    wanted = {"mobile_cr_drop_cause", "margin_question_unanswerable", "sessions_last_14d"}
    selected = [c for c in cases if c["id"] in wanted]
    assert len(selected) == 3
    for case in selected:
        outcome, _ = await run_case(eval_container, case)
        assert outcome.passed, f"{case['id']}: {outcome.failures}"


async def test_at_least_three_refusal_cases_pass(cases, eval_container):
    refusals = [c for c in cases if c["family"] == "refusal"]
    outcomes = [(await run_case(eval_container, c))[0] for c in refusals]
    assert sum(1 for o in outcomes if o.passed) >= 3


async def test_summary_reports_the_required_metrics(cases, eval_container):
    outcomes = [(await run_case(eval_container, c))[0] for c in cases[:4]]
    summary = summarize(outcomes)
    for key in ("pass_rate", "avg_cost_eur", "p50_latency_ms", "p95_latency_ms", "avg_tool_calls"):
        assert key in summary
