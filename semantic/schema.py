"""Modelo de datos canónico del sistema.

Todas las fronteras del sistema (conectores, tools, proveedor LLM, API) hablan
en estos tipos. Ningún módulo debe definir su propia versión de estas formas.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Window(StrEnum):
    LAST_7D = "last_7d"
    LAST_14D = "last_14d"
    LAST_30D = "last_30d"
    LAST_30d = "last_30d"  # alias tolerado: mismo valor, miembro alias
    PREV_7D = "prev_7d"
    PREV_14D = "prev_14d"
    PREV_30D = "prev_30d"
    SAME_PERIOD_LAST_YEAR = "same_period_last_year"


class Device(StrEnum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    ALL = "all"


class DataQuality(StrEnum):
    EXACT = "exact"  # dato completo de la fuente
    SAMPLED = "sampled"  # GA4 ha muestreado
    PARTIAL = "partial"  # ventana incompleta (datos aún consolidándose)
    DERIVED = "derived"  # compuesto de >1 fuente


Unit = Literal["count", "ratio", "currency_eur", "seconds"]


class Observation(BaseModel):
    """Unidad atómica de dato. Es lo único que un conector puede devolver."""

    metric: str
    value: float
    unit: Unit
    window: Window
    window_start: date
    window_end: date
    dimensions: dict[str, str] = Field(default_factory=dict)
    source: str  # "ga4" | "shopify" | "support" | "derived"
    quality: DataQuality
    sample_size: int | None = None  # n de la observación; None si no aplica
    note: str | None = None  # p.ej. composición de una métrica derivada
    fetched_at: datetime


class ToolCall(BaseModel):
    id: str
    tool: str
    arguments: dict
    latency_ms: int
    ok: bool
    error: str | None = None
    observations: list[Observation] = Field(default_factory=list)


class Evidence(BaseModel):
    tool_call_id: str  # DEBE existir en la traza del mismo run
    metric: str
    value: float
    window: Window
    dimensions: dict[str, str] = Field(default_factory=dict)
    source: str
    quality: DataQuality


class ProposedAction(BaseModel):
    what: str
    effort: Literal["low", "medium", "high"]
    estimated_impact: str
    validation: str  # cómo se comprobaría que ha funcionado


class Insight(BaseModel):
    id: str
    run_id: str
    hypothesis: str
    causal_claim: bool  # True solo si el modelo afirma causa, no correlación
    confidence: Literal["high", "medium", "low"]
    evidence: list[Evidence] = Field(min_length=1)
    proposed_action: ProposedAction | None = None
    assumptions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)  # fuentes que no cuadran
    degraded: bool = False
    degraded_reason: str | None = None

    @field_validator("hypothesis")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hypothesis no puede estar vacía")
        return v


class Trace(BaseModel):
    run_id: str
    question: str
    persona: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_eur: float = 0.0
    latency_ms: int = 0
    model: str = ""
    prompt_version: str = ""
    started_at: datetime


__all__ = [
    "DataQuality",
    "Device",
    "Evidence",
    "Insight",
    "Observation",
    "ProposedAction",
    "ToolCall",
    "Trace",
    "Unit",
    "Window",
]
