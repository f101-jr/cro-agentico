"""Orquestador.

Bucle explícito, sin framework de agentes:

  1. Construir mensajes: system prompt (versionado) + esquemas de tools + pregunta
  2. Llamar al proveedor
  3. Si la respuesta contiene tool_calls: ejecutarlas EN PARALELO cuando sean
     independientes, registrarlas en la traza, devolver resultados y volver a 2
  4. Si es respuesta final: parsear JSON contra `Insight`
  5. Validación post-hoc. Si falla -> 1 reintento con el error como feedback
  6. Si falla el reintento -> degradar
  7. Persistir `Trace`. Devolver `Insight`

El orquestador nunca construye SQL ni llama a un conector directamente: solo
invoca tools registradas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from agent.guardrails import Budget, RunTimeout
from agent.personas import DEFAULT_PERSONA, apply_persona, load_persona, persona_prompt
from agent.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    text_block,
    tool_result_block,
    tool_use_block,
)
from agent.validators import apply_causal_rule, degrade, validate_insight
from semantic.schema import Insight, ToolCall, Trace, Window
from semantic.tools import ToolContext, ToolResult, execute_tool, tool_schemas
from storage.traces import TraceStore

log = logging.getLogger("cro.orchestrator")

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "orchestrator.md"
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def load_system_prompt(path: Path = PROMPT_PATH) -> tuple[str, str]:
    """Devuelve (versión, cuerpo). La versión se persiste en cada `Trace`."""
    raw = path.read_text(encoding="utf-8")
    version = "0"
    body = raw
    if raw.startswith("---"):
        _, _, rest = raw.partition("---\n")
        block, _, body = rest.partition("---\n")
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "version":
                version = value.strip()
    return version, body.strip()


@dataclass
class RunResult:
    run_id: str
    persona: str
    insights: list[Insight]  # proyectados a la persona
    canonical_insights: list[Insight]  # sin proyectar: base de la igualdad de cifras entre personas
    trace: Trace
    degraded: bool = False
    stop_reason: str | None = None
    validation_retries: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Orchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        context: ToolContext,
        *,
        store: TraceStore | None = None,
        budget_factory=Budget,
        max_tokens: int = 4096,
    ) -> None:
        self.provider = provider
        self.context = context
        self.store = store
        self.budget_factory = budget_factory
        self.max_tokens = max_tokens
        self.prompt_version, self.prompt_body = load_system_prompt()

    # ------------------------------------------------------------------ run --
    async def run(
        self,
        question: str,
        *,
        persona: str = DEFAULT_PERSONA,
        window: Window | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        load_persona(persona)  # valida la persona antes de gastar un token
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        budget = self.budget_factory()
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        trace = Trace(run_id=run_id, question=question, persona=persona, started_at=started_at,
                      model=getattr(self.provider, "model", "?"), prompt_version=self.prompt_version)
        system = self._system_prompt(persona, window)
        messages: list[LLMMessage] = [LLMMessage(role="user", content=[text_block(self._question_block(question, window))])]

        insights: list[Insight] = []
        stop_reason: str | None = None
        retries = 0

        try:
            while True:
                exhausted = budget.exhausted()
                if exhausted == "total_timeout":
                    raise RunTimeout(f"run {run_id} ha superado {budget.total_timeout_s}s")
                if exhausted:
                    stop_reason = exhausted
                    log.warning("run %s cortado por %s", run_id, exhausted)
                    break

                budget.note_iteration()
                response = await self._complete(system, messages, budget)
                budget.note_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
                trace.prompt_tokens += response.usage.prompt_tokens
                trace.completion_tokens += response.usage.completion_tokens

                if response.wants_tools:
                    calls = response.tool_calls[: budget.allowed_tool_calls(len(response.tool_calls))]
                    if not calls:
                        stop_reason = "max_tool_calls"
                        break
                    results = await self._execute_parallel(calls, trace, budget)
                    messages.append(LLMMessage(role="assistant", content=(
                        ([text_block(response.text)] if response.text else [])
                        + [tool_use_block(c) for c in calls]
                    )))
                    messages.append(LLMMessage(role="user", content=[
                        tool_result_block(call.id, json.dumps(result.for_model(call.id), ensure_ascii=False))
                        for call, result in zip(calls, results)
                    ]))
                    continue

                parsed, parse_error = self._parse_insights(response.text, run_id)
                if parse_error and retries < budget.schema_retries:
                    retries += 1
                    messages.append(LLMMessage(role="assistant", content=[text_block(response.text)]))
                    messages.append(LLMMessage(role="user", content=[text_block(
                        f"VALIDACIÓN: no he podido parsear tu respuesta ({parse_error}). Devuelve "
                        f"exclusivamente el objeto JSON descrito en la sección Salida."
                    )]))
                    continue
                if parse_error:
                    stop_reason = "schema_error"
                    break

                insights, retries, revalidated = await self._validate_loop(
                    parsed, trace, messages, response, budget, retries, system
                )
                stop_reason = stop_reason or ("validation_degraded" if revalidated else None)
                break
        finally:
            trace.iterations = budget.iterations
            trace.latency_ms = int((time.perf_counter() - started) * 1000)
            trace.cost_eur = self.provider.cost_eur(_usage(trace))

        if not insights and stop_reason:
            fallback = self._degraded_fallback(run_id, trace, stop_reason)
            insights = [fallback] if fallback else []

        canonical = insights
        projected = apply_persona(canonical, persona, sample_sizes=_sample_sizes(trace))
        degraded = any(i.degraded for i in canonical)

        if self.store is not None:
            await self.store.save_run(trace, canonical)

        return RunResult(run_id=run_id, persona=persona, insights=projected, canonical_insights=canonical,
                         trace=trace, degraded=degraded, stop_reason=stop_reason, validation_retries=retries)

    # -------------------------------------------------------------- helpers --
    def _system_prompt(self, persona: str, window: Window | None) -> str:
        parts = [self.prompt_body, persona_prompt(persona)]
        if window is not None:
            parts.append(f"Ventana por defecto de esta consulta: `{window.value}`. Si la pregunta "
                         f"menciona otra, manda la de la pregunta.")
        return "\n\n---\n\n".join(parts)

    def _question_block(self, question: str, window: Window | None) -> str:
        if window is None:
            return question
        return f"{question}\n\n(ventana por defecto: {window.value})"

    async def _complete(self, system: str, messages: list[LLMMessage], budget: Budget) -> LLMResponse:
        timeout = max(budget.remaining_s(), 1.0)
        try:
            return await asyncio.wait_for(
                self.provider.complete(system=system, messages=messages, tools=tool_schemas(),
                                       max_tokens=self.max_tokens),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RunTimeout("el proveedor no ha respondido dentro del presupuesto del run") from exc

    async def _execute_parallel(
        self, calls: list[LLMToolCall], trace: Trace, budget: Budget
    ) -> list[ToolResult]:
        """Las tool_calls de una misma iteración se ejecutan concurrentemente."""
        budget.note_tool_calls(len(calls))

        async def _one(call: LLMToolCall) -> tuple[ToolResult, int, str | None]:
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    execute_tool(self.context, call.name, call.arguments),
                    timeout=budget.tool_timeout_s,
                )
                error = None if result.ok else (result.guidance or "tool sin datos")
            except asyncio.TimeoutError:
                result = ToolResult(tool=call.name, ok=False,
                                    guidance=f"{call.name} ha superado {budget.tool_timeout_s}s. "
                                             f"Prueba una ventana menor.")
                error = "tool_timeout"
            return result, int((time.perf_counter() - started) * 1000), error

        outcomes = await asyncio.gather(*(_one(call) for call in calls))
        results: list[ToolResult] = []
        for call, (result, latency_ms, error) in zip(calls, outcomes):
            trace.tool_calls.append(ToolCall(
                id=call.id, tool=call.name, arguments=call.arguments, latency_ms=latency_ms,
                ok=result.ok, error=error, observations=result.observations,
            ))
            results.append(result)
        return results

    def _parse_insights(self, text: str, run_id: str) -> tuple[list[Insight], str | None]:
        match = _JSON_BLOCK.search(text or "")
        if not match:
            return [], "la respuesta no contiene un objeto JSON"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return [], f"JSON inválido: {exc}"
        raw_insights = payload.get("insights", payload if isinstance(payload, list) else [])
        if not isinstance(raw_insights, list) or not raw_insights:
            return [], "el JSON no contiene una lista 'insights' con al menos un elemento"
        out: list[Insight] = []
        for i, raw in enumerate(raw_insights, start=1):
            raw = dict(raw)
            raw.setdefault("id", f"{run_id}-{i}")
            raw["run_id"] = run_id
            try:
                out.append(Insight.model_validate(raw))
            except ValidationError as exc:
                return [], f"insight {i} no cumple el esquema: {exc.errors()[:2]}"
        return out, None

    async def _validate_loop(
        self, parsed: list[Insight], trace: Trace, messages: list[LLMMessage],
        response: LLMResponse, budget: Budget, retries: int, system: str,
    ) -> tuple[list[Insight], int, bool]:
        """Validación post-hoc con un reintento; si vuelve a fallar, degrada."""
        insights = [apply_causal_rule(i)[0] for i in parsed]
        reports = [validate_insight(i, trace.tool_calls) for i in insights]
        if all(r.ok for r in reports):
            return insights, retries, False

        if retries < budget.schema_retries:
            retries += 1
            feedback = "\n\n".join(r.feedback() for r in reports if not r.ok)
            messages.append(LLMMessage(role="assistant", content=[text_block(response.text)]))
            messages.append(LLMMessage(role="user", content=[text_block(feedback)]))
            budget.note_iteration()
            retry_response = await self._complete(system, messages, budget)
            budget.note_tokens(retry_response.usage.prompt_tokens, retry_response.usage.completion_tokens)
            trace.prompt_tokens += retry_response.usage.prompt_tokens
            trace.completion_tokens += retry_response.usage.completion_tokens
            retried, error = self._parse_insights(retry_response.text, trace.run_id)
            if not error and retried:
                insights = [apply_causal_rule(i)[0] for i in retried]
                reports = [validate_insight(i, trace.tool_calls) for i in insights]
                if all(r.ok for r in reports):
                    return insights, retries, False

        degraded_insights = [
            insight if report.ok else degrade(insight, report)
            for insight, report in zip(insights, reports)
        ]
        return degraded_insights, retries, True

    def _degraded_fallback(self, run_id: str, trace: Trace, stop_reason: str) -> Insight | None:
        """Un corte de guardarraíl devuelve insight degradado, no error."""
        evidence = None
        for call in trace.tool_calls:
            for obs in call.observations:
                evidence = {
                    "tool_call_id": call.id, "metric": obs.metric, "value": obs.value,
                    "window": obs.window, "dimensions": obs.dimensions, "source": obs.source,
                    "quality": obs.quality,
                }
                break
            if evidence:
                break
        if evidence is None:
            return None
        motivo = {
            "max_iterations": "he agotado el número máximo de iteraciones",
            "max_tool_calls": "he agotado el número máximo de llamadas a tools",
            "token_budget": "he agotado el presupuesto de tokens del run",
            "schema_error": "no he conseguido producir una respuesta con el formato exigido",
            "total_timeout": "he agotado el tiempo del run",
        }.get(stop_reason, stop_reason)
        return Insight.model_validate({
            "id": f"{run_id}-1", "run_id": run_id,
            "hypothesis": (f"No he cerrado el análisis: {motivo}. Lo recogido hasta aquí no basta para "
                           f"sostener una conclusión, así que no la doy."),
            "causal_claim": False, "confidence": "low", "evidence": [evidence],
            "assumptions": [f"Run interrumpido por guardarraíl: {stop_reason}."],
            "degraded": True, "degraded_reason": stop_reason,
        })


def _usage(trace: Trace):
    from agent.providers.base import LLMUsage

    return LLMUsage(prompt_tokens=trace.prompt_tokens, completion_tokens=trace.completion_tokens)


def _sample_sizes(trace: Trace) -> dict[tuple[str, str], int]:
    sizes: dict[tuple[str, str], int] = {}
    for call in trace.tool_calls:
        for obs in call.observations:
            if obs.sample_size:
                sizes.setdefault((call.id, obs.metric), obs.sample_size)
    return sizes


__all__ = ["Orchestrator", "RunResult", "load_system_prompt"]
