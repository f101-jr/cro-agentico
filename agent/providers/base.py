"""Interfaz del proveedor LLM.

Ningún módulo fuera de `agent/providers/` importa el SDK de un proveedor. El
orquestador solo conoce estos tipos.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class LLMMessage(BaseModel):
    """Mensaje normalizado. `content` es una lista de bloques tipados:

    - {"type": "text", "text": ...}
    - {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
    - {"type": "tool_result", "tool_use_id": ..., "content": "json"}
    """

    role: Role
    content: list[dict[str, Any]]


class LLMToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = ""
    stop_reason: str = "end_turn"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    def cost_eur(self, usage: LLMUsage) -> float: ...


# Precio por millón de tokens, en EUR. Se declara aquí para que el coste de un
# run sea un dato de la traza y no una estimación posterior.
PRICING_EUR_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.76, 13.80),
    "claude-sonnet-4-5": (2.76, 13.80),
    "claude-opus-5": (13.80, 69.00),
    "claude-haiku-4-5": (0.74, 3.68),
    "scripted-offline": (0.0, 0.0),
}


def estimate_cost_eur(model: str, usage: LLMUsage) -> float:
    for prefix, (price_in, price_out) in PRICING_EUR_PER_MTOK.items():
        if model.startswith(prefix):
            return round(
                usage.prompt_tokens / 1_000_000 * price_in
                + usage.completion_tokens / 1_000_000 * price_out,
                6,
            )
    return 0.0


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_result_block(tool_use_id: str, payload: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": payload}


def tool_use_block(call: LLMToolCall) -> dict:
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}


__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMToolCall",
    "LLMUsage",
    "PRICING_EUR_PER_MTOK",
    "estimate_cost_eur",
    "text_block",
    "tool_result_block",
    "tool_use_block",
]
