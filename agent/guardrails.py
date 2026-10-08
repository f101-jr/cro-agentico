"""Guardarraíles del run.

Son cortes en seco, no sugerencias: un run que los supera se degrada, nunca se
alarga “un poco más”.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_ITERATIONS = 8
TOKEN_BUDGET = 40_000  # por run, corte en seco
TOOL_TIMEOUT_S = 15
TOTAL_TIMEOUT_S = 90
SCHEMA_RETRIES = 1
MAX_TOOL_CALLS = 20  # por run


class RunTimeout(RuntimeError):
    """Se ha superado TOTAL_TIMEOUT_S. La API lo traduce a 504."""


@dataclass
class Budget:
    """Contador vivo del run. El orquestador le pregunta antes de cada paso."""

    max_iterations: int = MAX_ITERATIONS
    token_budget: int = TOKEN_BUDGET
    max_tool_calls: int = MAX_TOOL_CALLS
    total_timeout_s: float = TOTAL_TIMEOUT_S
    tool_timeout_s: float = TOOL_TIMEOUT_S
    schema_retries: int = SCHEMA_RETRIES

    iterations: int = 0
    tokens: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    stop_reason: str | None = None

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_s(self) -> float:
        return max(self.total_timeout_s - self.elapsed_s(), 0.0)

    def note_iteration(self) -> None:
        self.iterations += 1

    def note_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.tokens += prompt_tokens + completion_tokens

    def note_tool_calls(self, n: int) -> None:
        self.tool_calls += n

    def exhausted(self) -> str | None:
        """Devuelve el motivo de corte, o None si el run puede continuar."""
        if self.iterations >= self.max_iterations:
            return "max_iterations"
        if self.tokens >= self.token_budget:
            return "token_budget"
        if self.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.elapsed_s() >= self.total_timeout_s:
            return "total_timeout"
        return None

    def allowed_tool_calls(self, requested: int) -> int:
        return max(min(requested, self.max_tool_calls - self.tool_calls), 0)


__all__ = [
    "Budget",
    "MAX_ITERATIONS",
    "MAX_TOOL_CALLS",
    "RunTimeout",
    "SCHEMA_RETRIES",
    "TOKEN_BUDGET",
    "TOOL_TIMEOUT_S",
    "TOTAL_TIMEOUT_S",
]
