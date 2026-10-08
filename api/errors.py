"""Formato único de error de la API.

Un run degradado NO es un error: devuelve 200 con `degraded=true`. Aquí solo
viven los fallos de verdad.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    run_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    """Error con código de dominio. El HTTP lo decide la tabla de errores."""

    def __init__(self, status_code: int, code: str, message: str, run_id: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.run_id = run_id

    def to_response(self, run_id: str | None = None) -> JSONResponse:
        body = ErrorResponse(error=ErrorBody(code=self.code, message=self.message,
                                             run_id=self.run_id or run_id))
        return JSONResponse(status_code=self.status_code, content=body.model_dump(),
                            headers={"X-Run-Id": self.run_id or run_id or "-"})


def invalid_persona(value: str, valid: tuple[str, ...]) -> ApiError:
    return ApiError(400, "invalid_persona",
                    f"persona '{value}' no válida. Válidas: {', '.join(valid)}.")


def invalid_window(value: str, valid: list[str]) -> ApiError:
    return ApiError(400, "invalid_window",
                    f"window '{value}' no válida. Válidas: {', '.join(valid)}.")


def insight_not_found(insight_id: str) -> ApiError:
    return ApiError(404, "insight_not_found", f"no existe el insight '{insight_id}'.")


def run_not_found(run_id: str) -> ApiError:
    return ApiError(404, "run_not_found", f"no existe el run '{run_id}'.", run_id=run_id)


def question_too_long(length: int, limit: int) -> ApiError:
    return ApiError(422, "question_too_long",
                    f"la pregunta tiene {length} caracteres y el límite es {limit}.")


def source_unavailable(sources: list[str], run_id: str | None = None) -> ApiError:
    return ApiError(424, "source_unavailable",
                    f"fuentes obligatorias sin respuesta tras los reintentos: {', '.join(sources)}.",
                    run_id=run_id)


def rate_limited(limit: int) -> ApiError:
    return ApiError(429, "rate_limited", f"límite propio de {limit} peticiones por minuto superado.")


def run_timeout(seconds: float, run_id: str | None = None) -> ApiError:
    return ApiError(504, "run_timeout", f"el run ha superado {seconds:.0f}s.", run_id=run_id)


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return exc.to_response(getattr(request.state, "run_id", None))


__all__ = [
    "ApiError",
    "ErrorBody",
    "ErrorResponse",
    "api_error_handler",
    "insight_not_found",
    "invalid_persona",
    "invalid_window",
    "question_too_long",
    "rate_limited",
    "run_not_found",
    "run_timeout",
    "source_unavailable",
]
