"""Runner de evals.

Ejecuta el set dorado contra fixtures, **sin red**, y reporta:
% de casos superados, coste medio por run, p50/p95 de latencia y media de
tool_calls por run.

Uso:
    uv run python -m evals.run                  # set completo
    uv run python -m evals.run --case mobile_cr_drop_cause
    uv run python -m evals.run --family refusal --verbose
    uv run python -m evals.run --json evals/reports/last.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.orchestrator import RunResult
from agent.validators import _matches, extract_figures, validate_insight
from config import Settings
from container import build_container
from semantic.schema import Insight, Window

GOLDEN = Path(__file__).resolve().parent / "golden.yaml"
FAMILIES = ("causal", "refusal", "factual")


def _norm(text: str) -> str:
    lowered = (text or "").casefold()
    return "".join(c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn")


def insight_text(insights: list[Insight]) -> str:
    chunks: list[str] = []
    for insight in insights:
        chunks.append(insight.hypothesis)
        chunks.extend(insight.assumptions)
        chunks.extend(insight.contradictions)
        if insight.proposed_action:
            action = insight.proposed_action
            chunks += [action.what, action.estimated_impact, action.validation]
    return "\n".join(chunks)


@dataclass
class CaseOutcome:
    id: str
    family: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    latency_ms: int = 0
    cost_eur: float = 0.0
    tool_calls: int = 0
    iterations: int = 0
    degraded: bool = False


def check_case(case: dict, result: RunResult) -> list[str]:
    failures: list[str] = []
    insights = result.insights
    trace = result.trace
    text = _norm(insight_text(insights))
    tools_called = [call.tool for call in trace.tool_calls]
    sources = {e.source for i in insights for e in i.evidence}

    if not insights:
        failures.append("el run no ha devuelto ningún insight")

    for tool in case.get("must_call_tools", []):
        if tool not in tools_called:
            failures.append(f"no se ha llamado a {tool} (llamadas: {sorted(set(tools_called))})")

    for source in case.get("must_cite_sources", []):
        if source not in sources:
            failures.append(f"ninguna evidencia cita la fuente '{source}' (cita: {sorted(sources)})")

    for needle in case.get("expect_contains", []):
        if _norm(needle) not in text:
            failures.append(f"el texto no contiene {needle!r}")

    for needle in case.get("forbid_contains", []):
        if _norm(needle) in text:
            failures.append(f"el texto contiene {needle!r} y no debería")

    if "expect_causal_claim" in case:
        got = any(i.causal_claim for i in insights)
        if got is not bool(case["expect_causal_claim"]):
            failures.append(f"causal_claim={got}, esperado {case['expect_causal_claim']}")

    if "expect_figure" in case:
        wanted = float(case["expect_figure"])
        if not _figure_present(wanted, insights, text):
            failures.append(f"la cifra {wanted} no aparece en la respuesta")

    if "max_tool_calls" in case and len(trace.tool_calls) > int(case["max_tool_calls"]):
        failures.append(f"{len(trace.tool_calls)} tool_calls, máximo {case['max_tool_calls']}")

    if "min_evidence" in case:
        worst = min((len(i.evidence) for i in insights), default=0)
        if worst < int(case["min_evidence"]):
            failures.append(f"un insight cita {worst} evidencias, mínimo {case['min_evidence']}")

    if case.get("expect_degraded_or_assumption"):
        if not any(i.degraded or i.assumptions for i in insights):
            failures.append("ningún insight está degradado ni declara supuestos")

    if case.get("expect_degraded"):
        if not any(i.degraded for i in insights):
            failures.append("ningún insight está degradado")

    if case.get("forbid_figures_not_in_tools", False):
        for insight in insights:
            report = validate_insight(insight, trace.tool_calls)
            if report.orphan_figures:
                failures.append(f"cifras sin respaldo en tools: {report.orphan_figures}")

    if "max_insights" in case and len(insights) > int(case["max_insights"]):
        failures.append(f"{len(insights)} insights, máximo {case['max_insights']}")

    return failures


def _figure_present(wanted: float, insights: list[Insight], text: str) -> bool:
    for _, candidates in extract_figures(text):
        for candidate in candidates:
            if abs(candidate - wanted) <= max(abs(wanted) * 1e-4, 0.01):
                return True
    for insight in insights:
        for evidence in insight.evidence:
            if _matches(wanted, evidence.value, "count") or _matches(wanted, evidence.value, "ratio"):
                return True
    return False


async def run_case(container, case: dict) -> tuple[CaseOutcome, RunResult | None]:
    window = Window(case["window"]) if case.get("window") else None
    persona = case.get("persona", "analyst")
    try:
        result = await container.orchestrator.run(case["question"], persona=persona, window=window)
    except Exception as exc:  # noqa: BLE001 - un caso que revienta es un caso fallado
        return CaseOutcome(id=case["id"], family=case.get("family", "?"), passed=False,
                           failures=[f"excepción: {type(exc).__name__}: {exc}"]), None
    failures = check_case(case, result)
    return CaseOutcome(
        id=case["id"], family=case.get("family", "?"), passed=not failures, failures=failures,
        latency_ms=result.trace.latency_ms, cost_eur=result.trace.cost_eur,
        tool_calls=len(result.trace.tool_calls), iterations=result.trace.iterations,
        degraded=result.degraded,
    ), result


def load_cases(path: Path = GOLDEN) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("golden.yaml debe ser una lista de casos")
    return cases


def summarize(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    latencies = sorted(o.latency_ms for o in outcomes) or [0]
    passed = sum(1 for o in outcomes if o.passed)
    return {
        "cases": len(outcomes),
        "passed": passed,
        "pass_rate": round(passed / len(outcomes), 4) if outcomes else 0.0,
        "avg_cost_eur": round(statistics.fmean(o.cost_eur for o in outcomes), 6) if outcomes else 0.0,
        "p50_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        "avg_tool_calls": round(statistics.fmean(o.tool_calls for o in outcomes), 2) if outcomes else 0.0,
        "avg_iterations": round(statistics.fmean(o.iterations for o in outcomes), 2) if outcomes else 0.0,
        "degraded_runs": sum(1 for o in outcomes if o.degraded),
        "by_family": {
            family: {
                "cases": len([o for o in outcomes if o.family == family]),
                "passed": len([o for o in outcomes if o.family == family and o.passed]),
            }
            for family in FAMILIES
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.golden))
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    if args.family:
        cases = [c for c in cases if c.get("family") in args.family]
    if not cases:
        print("no hay casos que ejecutar", file=sys.stderr)
        return 2

    # Fixtures y proveedor determinista: el set completo corre sin red.
    container = build_container(Settings(mode="fixtures", llm_provider="scripted", trace_db=":memory:"))

    outcomes: list[CaseOutcome] = []
    for case in cases:
        outcome, result = await run_case(container, case)
        outcomes.append(outcome)
        mark = "OK  " if outcome.passed else "FALLA"
        print(f"{mark} {outcome.id:<34} {outcome.family:<8} "
              f"{outcome.tool_calls:>2} tools · {outcome.latency_ms:>5} ms")
        for failure in outcome.failures:
            print(f"      - {failure}")
        if args.verbose and result is not None:
            for insight in result.insights:
                print(f"      > {insight.hypothesis[:300]}")

    summary = summarize(outcomes)
    print("\n" + "=" * 72)
    print(f"casos superados: {summary['passed']}/{summary['cases']} ({summary['pass_rate']:.0%})")
    print(f"coste medio por run: {summary['avg_cost_eur']:.6f} €")
    print(f"latencia p50/p95: {summary['p50_latency_ms']} ms / {summary['p95_latency_ms']} ms")
    print(f"tool_calls por run: {summary['avg_tool_calls']} · iteraciones: {summary['avg_iterations']}")
    print(f"runs degradados: {summary['degraded_runs']}")
    for family, data in summary["by_family"].items():
        print(f"  {family:<8} {data['passed']}/{data['cases']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "summary": summary,
            "cases": [o.__dict__ for o in outcomes],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ninforme escrito en {out}")

    return 0 if summary["passed"] == summary["cases"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner del set dorado de evals")
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--case", action="append", help="ejecuta solo estos ids")
    parser.add_argument("--family", action="append", choices=list(FAMILIES))
    parser.add_argument("--json", help="ruta del informe JSON")
    parser.add_argument("--verbose", action="store_true")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
