"""API HTTP.

Devuelve JSON. No renderiza nada: el frontend es de otro.
Todas las respuestas llevan la cabecera `X-Run-Id`.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from pydantic import BaseModel, Field

from agent.guardrails import TOTAL_TIMEOUT_S, RunTimeout
from agent.orchestrator import RunResult
from agent.personas import DEFAULT_PERSONA, PERSONAS, apply_persona
from api import errors
from api.errors import ApiError, api_error_handler
from config import Settings
from connectors.base import HealthStatus
from container import Container, build_container
from semantic.schema import Evidence, Insight, Observation, Trace, Window
from storage.traces import observations_for

MAX_QUESTION_CHARS = 500
DASHBOARD_QUESTION = (
    "¿Qué ha cambiado en la conversión, los pedidos y los ingresos en esta ventana, y por qué?"
)


# --------------------------------------------------------------------------- #
# Modelos de respuesta
# --------------------------------------------------------------------------- #
class InsightsResponse(BaseModel):
    run_id: str
    persona: str
    generated_at: datetime
    window: Window
    degraded: bool = False
    insights: list[Insight] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    persona: str = DEFAULT_PERSONA
    window: Window | None = None


class AskResponse(BaseModel):
    run_id: str
    persona: str
    generated_at: datetime
    degraded: bool = False
    insights: list[Insight] = Field(default_factory=list)


class EvidenceResponse(BaseModel):
    insight_id: str
    run_id: str
    evidence: list[Evidence]
    observations: list[Observation]


class HealthResponse(BaseModel):
    sources: list[HealthStatus]


# --------------------------------------------------------------------------- #
# Límite propio de peticiones
# --------------------------------------------------------------------------- #
class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def check(self, client: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        window = self._hits.setdefault(client, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            raise errors.rate_limited(self.per_minute)
        window.append(now)


def get_container(request: Request) -> Container:
    """Dependencia común: aplica el límite propio antes de servir nada."""
    request.app.state.limiter.check(request.client.host if request.client else "anonymous")
    return request.app.state.container


Ctn = Annotated[Container, Depends(get_container)]


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #
def create_app(container: Container | None = None, *, settings: Settings | None = None) -> FastAPI:
    container = container or build_container(settings)
    app = FastAPI(
        title="CRO agéntico",
        version="0.1.0",
        summary="Responde preguntas de negocio cruzando GA4, Shopify y soporte. Devuelve JSON.",
    )
    app.state.container = container
    app.state.limiter = RateLimiter(container.settings.rate_limit_per_minute)
    app.add_exception_handler(ApiError, api_error_handler)

    @app.middleware("http")
    async def run_id_header(request: Request, call_next):
        request.state.run_id = f"req_{uuid.uuid4().hex[:12]}"
        response: Response = await call_next(request)
        response.headers.setdefault("X-Run-Id", request.state.run_id)
        return response

    # ---------------------------------------------------------------- rutas --
    @app.get("/insights", response_model=InsightsResponse, responses=_error_responses())
    async def get_insights(request: Request, response: Response, container: Ctn,
                           persona: str = DEFAULT_PERSONA, window: str | None = None) -> InsightsResponse:
        persona = _validate_persona(persona)
        win = _validate_window(window) or container.settings.default_window
        result = await _run(container, DASHBOARD_QUESTION, persona=persona, window=win)
        _stamp(request, response, result.run_id)
        return InsightsResponse(run_id=result.run_id, persona=persona, window=win,
                                generated_at=result.generated_at, degraded=result.degraded,
                                insights=result.insights)

    @app.post("/ask", response_model=AskResponse, responses=_error_responses())
    async def ask(request: Request, response: Response, container: Ctn, body: AskRequest) -> AskResponse:
        if len(body.question) > MAX_QUESTION_CHARS:
            raise errors.question_too_long(len(body.question), MAX_QUESTION_CHARS)
        persona = _validate_persona(body.persona)
        result = await _run(container, body.question, persona=persona, window=body.window)
        _stamp(request, response, result.run_id)
        return AskResponse(run_id=result.run_id, persona=persona, generated_at=result.generated_at,
                           degraded=result.degraded, insights=result.insights)

    @app.get("/evidence/{insight_id}", response_model=EvidenceResponse, responses=_error_responses())
    async def get_evidence(request: Request, response: Response, container: Ctn,
                           insight_id: str) -> EvidenceResponse:
        found = await container.store.find_insight(insight_id)
        if found is None:
            raise errors.insight_not_found(insight_id)
        insight, trace = found
        _stamp(request, response, trace.run_id)
        return EvidenceResponse(insight_id=insight.id, run_id=trace.run_id, evidence=insight.evidence,
                                observations=observations_for(trace, insight))

    @app.get("/trace/{run_id}", response_model=Trace, responses=_error_responses())
    async def get_trace(request: Request, response: Response, container: Ctn, run_id: str) -> Trace:
        trace = await container.store.get_trace(run_id)
        if trace is None:
            raise errors.run_not_found(run_id)
        _stamp(request, response, run_id)
        return trace

    @app.get("/connectors/health", response_model=HealthResponse, responses=_error_responses())
    async def connectors_health(request: Request, response: Response, container: Ctn) -> HealthResponse:
        from semantic.tools import execute_tool

        result = await execute_tool(container.context, "get_data_health", {}, use_cache=False)
        _stamp(request, response, request.state.run_id)
        return HealthResponse(sources=result.health)

    return app


# --------------------------------------------------------------------------- #
# Piezas compartidas
# --------------------------------------------------------------------------- #
async def _run(container: Container, question: str, *, persona: str, window: Window | None) -> RunResult:
    try:
        result = await container.orchestrator.run(question, persona=persona, window=window)
    except RunTimeout as exc:
        raise errors.run_timeout(TOTAL_TIMEOUT_S) from exc

    if result.trace.tool_calls and not any(call.ok for call in result.trace.tool_calls):
        from semantic.tools import execute_tool

        health = await execute_tool(container.context, "get_data_health", {}, use_cache=False)
        down = [h.source for h in health.health if not h.reachable]
        if down:
            raise errors.source_unavailable(down, run_id=result.run_id)
    return result


def _validate_persona(value: str) -> str:
    if value not in PERSONAS:
        raise errors.invalid_persona(value, PERSONAS)
    return value


def _validate_window(value: str | None) -> Window | None:
    if value in (None, ""):
        return None
    try:
        return Window(value)
    except ValueError as exc:
        raise errors.invalid_window(str(value), [w.value for w in Window]) from exc


def _stamp(request: Request, response: Response, run_id: str) -> None:
    request.state.run_id = run_id
    response.headers["X-Run-Id"] = run_id


def _error_responses() -> dict:
    from api.errors import ErrorResponse

    return {code: {"model": ErrorResponse} for code in (400, 404, 422, 424, 429, 504)}


app = create_app


def render_for_persona(result: RunResult, persona: str) -> list[Insight]:
    """Renderiza un run ya ejecutado con otra persona: mismas cifras, por construcción."""
    from agent.orchestrator import _sample_sizes

    return apply_persona(result.canonical_insights, persona, sample_sizes=_sample_sizes(result.trace))


__all__ = ["AskRequest", "AskResponse", "DASHBOARD_QUESTION", "EvidenceResponse", "InsightsResponse",
           "MAX_QUESTION_CHARS", "create_app", "render_for_persona"]
