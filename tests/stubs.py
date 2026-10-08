"""Proveedores de mentira para los tests del orquestador."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent.providers.base import LLMMessage, LLMResponse, LLMToolCall, LLMUsage


@dataclass
class StubProvider:
    """Devuelve respuestas predefinidas, una por llamada."""

    responses: list[LLMResponse] = field(default_factory=list)
    model: str = "stub-model"
    calls: int = 0
    seen_systems: list[str] = field(default_factory=list)

    async def complete(self, *, system, messages, tools, max_tokens=4096) -> LLMResponse:
        self.seen_systems.append(system)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response

    def cost_eur(self, usage: LLMUsage) -> float:
        return round(usage.prompt_tokens * 1e-6 + usage.completion_tokens * 5e-6, 6)


def tool_response(*calls: tuple[str, dict]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[LLMToolCall(id=f"call_{i}", name=name, arguments=args)
                    for i, (name, args) in enumerate(calls)],
        stop_reason="tool_use", model="stub-model",
        usage=LLMUsage(prompt_tokens=100, completion_tokens=20),
    )


def json_response(payload: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload, ensure_ascii=False), stop_reason="end_turn",
                       model="stub-model", usage=LLMUsage(prompt_tokens=100, completion_tokens=50))


def insight_payload(hypothesis: str, evidence: list[dict], *, causal: bool = False,
                    confidence: str = "medium", assumptions: list[str] | None = None) -> dict:
    return {"insights": [{
        "hypothesis": hypothesis,
        "causal_claim": causal,
        "confidence": confidence,
        "evidence": evidence,
        "proposed_action": None,
        "assumptions": assumptions or [],
        "contradictions": [],
    }]}


def evidence_from(trace_or_result, metric: str, *, tool_call_id: str | None = None) -> dict:
    """Construye una evidencia válida a partir de la primera observación que cuadre."""
    trace = getattr(trace_or_result, "trace", trace_or_result)
    for call in trace.tool_calls:
        for obs in call.observations:
            if obs.metric == metric:
                return {"tool_call_id": tool_call_id or call.id, "metric": obs.metric,
                        "value": obs.value,
                        "window": obs.window.value, "dimensions": obs.dimensions,
                        "source": obs.source, "quality": obs.quality.value}
    raise AssertionError(f"no hay observación de {metric} en la traza")
