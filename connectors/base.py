"""Contrato común de los conectores.

Todo acceso a fuentes es de SOLO LECTURA. Ningún conector expone escritura, ni
siquiera como método privado.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from semantic.schema import Device, Observation, Window

log = logging.getLogger("cro.connectors")


class Capability(BaseModel):
    metric: str
    dimensions: list[str] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    min_granularity: Literal["day", "hour"] = "day"


class HealthStatus(BaseModel):
    source: str
    reachable: bool
    last_complete_date: date | None = None  # hasta qué día hay dato consolidado
    latency_ms: int | None = None
    message: str | None = None


class FetchRequest(BaseModel):
    """Petición canónica a un conector. La emite la capa semántica, nunca el LLM."""

    metric: str
    window: Window
    device: Device = Device.ALL
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int | None = None


@runtime_checkable
class Connector(Protocol):
    name: str

    def capabilities(self) -> list[Capability]: ...
    async def fetch(self, request: FetchRequest) -> list[Observation]: ...
    async def health(self) -> HealthStatus: ...


class SourceUnavailableError(RuntimeError):
    """Una fuente obligatoria no responde tras agotar los reintentos."""

    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message


class TransientSourceError(RuntimeError):
    """Error recuperable de la fuente (429 / 503). Dispara reintento."""

    def __init__(self, source: str, status: int, message: str = "") -> None:
        super().__init__(f"[{source}] {status} {message}".strip())
        self.source = source
        self.status = status


MAX_ATTEMPTS = 3
BASE_BACKOFF_S = 0.5


async def with_retries(
    fn: Callable[[], Awaitable[Any]],
    *,
    source: str,
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_BACKOFF_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Backoff exponencial ante errores transitorios (429/503).

    Agotados los intentos, la excepción que sale es `SourceUnavailableError`:
    la capa semántica la traduce a texto orientativo y la API a 424.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except TransientSourceError as exc:
            last = exc
            log.warning("fuente %s transitoria (%s), intento %d/%d", source, exc.status, attempt + 1, attempts)
            if attempt + 1 < attempts:
                await sleep(base_delay * (2**attempt))
    raise SourceUnavailableError(source, f"sin respuesta tras {attempts} intentos: {last}")


# --------------------------------------------------------------------------- #
# PII: se descarta EN EL MAPEO a Observation, antes de cualquier log
# --------------------------------------------------------------------------- #
PII_FIELDS = frozenset({
    "email", "customer_email", "contact_email", "name", "first_name", "last_name",
    "customer_name", "full_name", "phone", "telephone", "customer_phone", "address",
    "address1", "address2", "shipping_address", "billing_address", "customer_id",
    "customer", "zip", "postal_code", "ip", "client_id", "user_id",
})

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{2,3}[ .-]?)?(?:\d[ .-]?){8,12}\d(?!\d)")


def strip_pii(record: dict) -> dict:
    """Devuelve una copia del registro sin PII, en profundidad.

    Es el único punto por el que un dato de fuente entra al sistema: si un campo
    no se reconoce pero huele a email o a teléfono, también cae.
    """
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k.lower() not in PII_FIELDS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and (EMAIL_RE.search(value) or _looks_like_phone(value)):
            return "[redacted]"
        return value

    return clean(record)


def _looks_like_phone(value: str) -> bool:
    digits = sum(c.isdigit() for c in value)
    return digits >= 9 and bool(PHONE_RE.search(value))


def contains_pii(payload: Any) -> list[str]:
    """Busca patrones de PII en una estructura ya serializada. Lo usan los tests."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() in PII_FIELDS:
                    found.append(f"{path}.{k}: campo PII")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            if EMAIL_RE.search(node):
                found.append(f"{path}: email")
            elif _looks_like_phone(node):
                found.append(f"{path}: teléfono")

    walk(payload, "$")
    return found


__all__ = [
    "BASE_BACKOFF_S",
    "Capability",
    "Connector",
    "FetchRequest",
    "HealthStatus",
    "MAX_ATTEMPTS",
    "SourceUnavailableError",
    "TransientSourceError",
    "contains_pii",
    "strip_pii",
    "with_retries",
]
