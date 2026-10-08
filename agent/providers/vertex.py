"""Claude vía Vertex AI: región UE y retención cero.

Es el ÚNICO módulo que importa el SDK del proveedor. El import es perezoso para
que el sistema arranque, se testee y ejecute evals sin la dependencia instalada.
"""

from __future__ import annotations

import os
from typing import Any

from agent.providers.base import LLMMessage, LLMResponse, LLMToolCall, LLMUsage, estimate_cost_eur

DEFAULT_MODEL = "claude-sonnet-5@20250929"
DEFAULT_REGION = "europe-west1"  # región UE


class VertexClaudeProvider:
    def __init__(
        self,
        *,
        model: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or os.getenv("CRO_VERTEX_MODEL", DEFAULT_MODEL)
        self.project_id = project_id or os.getenv("CRO_VERTEX_PROJECT")
        self.region = region or os.getenv("CRO_VERTEX_REGION", DEFAULT_REGION)
        self._client = client

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropicVertex  # import perezoso: dependencia opcional

            if not self.project_id:
                raise RuntimeError("CRO_VERTEX_PROJECT no está configurado")
            self._client = AsyncAnthropicVertex(project_id=self.project_id, region=self.region)
        return self._client

    async def complete(
        self, *, system: str, messages: list[LLMMessage], tools: list[dict], max_tokens: int = 4096
    ) -> LLMResponse:
        response = await self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[m.model_dump() for m in messages],
            tools=[
                {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                for t in tools
            ],
        )
        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(LLMToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            usage=LLMUsage(prompt_tokens=response.usage.input_tokens,
                           completion_tokens=response.usage.output_tokens),
            model=self.model,
            stop_reason=response.stop_reason or "end_turn",
        )

    def cost_eur(self, usage: LLMUsage) -> float:
        return estimate_cost_eur(self.model.split("@")[0], usage)


__all__ = ["DEFAULT_MODEL", "DEFAULT_REGION", "VertexClaudeProvider"]
