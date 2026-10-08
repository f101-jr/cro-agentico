"""Conector de soporte.

Tercera fuente. En v1 sirve fixtures; la interfaz es la que tendría un Zendesk
real. Métrica: volumen de tickets por tema, con clasificación por palabras clave
**en código** — nunca por LLM: una clasificación no determinista convertiría el
control de calidad del dato en otra alucinación.

Del ticket solo sale el recuento: ni asunto ni cuerpo ni cliente cruzan la
frontera del conector.
"""

from __future__ import annotations

import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator, Protocol, runtime_checkable

from connectors.base import Capability, FetchRequest, HealthStatus, with_retries
from semantic.schema import DataQuality, Observation, Window

SOURCE = "support"
TOPICS = ("shipping", "payment", "product", "returns", "account", "other")

# El orden IMPORTA: "¿por qué me cobran envío?" es un ticket de envío, no de pago.
TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shipping", ("envio", "envios", "portes", "gastos de envio", "entrega", "mensajeria",
                  "transportista", "gratuito", "envio gratis")),
    ("returns", ("devolucion", "devoluciones", "devolver", "reembolso", "retorno")),
    ("payment", ("pago", "pagar", "tarjeta", "bizum", "paypal", "cobro", "cobrado", "factura")),
    ("account", ("cuenta", "contrasena", "acceder", "acceso", "login", "registro")),
    ("product", ("talla", "producto", "articulo", "stock", "defecto", "color", "material")),
)


def _normalize(text: str) -> str:
    lowered = text.casefold()
    return "".join(c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn")


def classify_topic(subject: str) -> str:
    """Clasificación determinista por palabras clave. Es código, no un modelo."""
    norm = _normalize(subject or "")
    for topic, keywords in TOPIC_KEYWORDS:
        if any(kw in norm for kw in keywords):
            return topic
    return "other"


@runtime_checkable
class SupportTransport(Protocol):
    async def iter_tickets(self, *, start: date, end: date) -> AsyncIterator[dict]: ...
    async def ping(self) -> date | None: ...


class SupportFixtureTransport:
    def __init__(self, *, timezone_name: str = "Europe/Madrid") -> None:
        from fixtures.dataset import load_dataset

        self._ds = load_dataset()
        self.timezone_name = timezone_name
        self.calls = 0

    async def iter_tickets(self, *, start: date, end: date) -> AsyncIterator[dict]:
        self.calls += 1
        today = _today(self.timezone_name)
        offsets = range((today - end).days, (today - start).days + 1)
        for t in self._ds.tickets_in(offsets):
            yield {"id": t["id"], "created_at": (today - timedelta(days=t["day_offset"])).isoformat(),
                   "subject": t["subject"], "channel": t["channel"]}

    async def ping(self) -> date | None:
        self.calls += 1
        return _today(self.timezone_name)


class SupportHttpTransport:
    """Forma que tendría un Zendesk real: búsqueda paginada por fecha de creación."""

    def __init__(self, base_url: str, token: str = "", *, client=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = client

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                timeout=20.0,
            )
        return self._client

    async def iter_tickets(self, *, start: date, end: date) -> AsyncIterator[dict]:
        page = 1
        while True:
            r = await self._http().get(
                "/api/v2/search.json",
                params={"query": f"type:ticket created>={start} created<={end}", "page": page},
            )
            r.raise_for_status()
            payload = r.json()
            for ticket in payload.get("results", []):
                yield {"id": str(ticket.get("id")), "created_at": ticket.get("created_at"),
                       "subject": ticket.get("subject", ""), "channel": ticket.get("via", {}).get("channel", "")}
            if not payload.get("next_page"):
                return
            page += 1

    async def ping(self) -> date | None:
        r = await self._http().get("/api/v2/tickets/count.json")
        r.raise_for_status()
        return _today()


class SupportConnector:
    name = SOURCE

    def __init__(self, transport: SupportTransport, *, retry_delay: float = 0.5,
                 timezone_name: str = "Europe/Madrid") -> None:
        self.transport = transport
        self._retry_delay = retry_delay
        self.timezone_name = timezone_name

    def capabilities(self) -> list[Capability]:
        return [Capability(metric="support_tickets", dimensions=["topic"], windows=list(Window))]

    def window_dates(self, window: Window) -> tuple[date, date]:
        from zoneinfo import ZoneInfo

        from connectors.windows import resolve_window

        return resolve_window(window, tz=ZoneInfo(self.timezone_name))

    async def fetch(self, request: FetchRequest) -> list[Observation]:
        start, end = self.window_dates(request.window)
        wanted = request.filters.get("topic", "all")

        async def _collect() -> dict[str, int]:
            counts = dict.fromkeys(TOPICS, 0)
            async for ticket in self.transport.iter_tickets(start=start, end=end):
                counts[classify_topic(ticket.get("subject", ""))] += 1
            return counts

        counts = await with_retries(_collect, source=SOURCE, base_delay=self._retry_delay)
        total = sum(counts.values())
        base = dict(window=request.window, window_start=start, window_end=end, source=SOURCE,
                    quality=DataQuality.EXACT, fetched_at=datetime.now(timezone.utc))
        note = ("Tema asignado por palabras clave en código sobre el asunto del ticket. "
                "Del ticket solo sale el recuento.")

        topics = TOPICS if wanted == "all" else (wanted,)
        obs = [
            Observation(metric="support_tickets", value=float(counts.get(topic, 0)), unit="count",
                        dimensions={"topic": topic}, sample_size=total, note=note, **base)
            for topic in topics
        ]
        if wanted == "all":
            obs.append(Observation(metric="support_tickets", value=float(total), unit="count",
                                   dimensions={"topic": "all"}, sample_size=total, note=note, **base))
        return obs

    async def health(self) -> HealthStatus:
        started = time.perf_counter()
        try:
            last_complete = await with_retries(self.transport.ping, source=SOURCE, base_delay=self._retry_delay)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(source=SOURCE, reachable=False,
                                latency_ms=int((time.perf_counter() - started) * 1000), message=str(exc))
        return HealthStatus(source=SOURCE, reachable=True, last_complete_date=last_complete,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            message="tickets clasificados por palabras clave en código")


def _today(tz_name: str = "Europe/Madrid") -> date:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(tz_name)).date()


__all__ = ["SupportConnector", "SupportFixtureTransport", "SupportHttpTransport", "TOPICS", "classify_topic"]
