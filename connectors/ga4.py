"""Conector GA4.

Separación estricta en dos capas:

  * **transporte** — habla con la Google Analytics Data API v1beta (o con los
    fixtures). Es lo único que `CRO_MODE=fixtures` sustituye.
  * **semántica** — resuelve ventanas a fechas en la zona horaria de la
    propiedad, decide la calidad del dato y mapea a `Observation`.

Scopes requeridos (solo lectura): `https://www.googleapis.com/auth/analytics.readonly`.
Credenciales vía Application Default Credentials; nunca en el repo.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from connectors.base import (
    Capability,
    FetchRequest,
    HealthStatus,
    TransientSourceError,
    strip_pii,
    with_retries,
)
from connectors.windows import day_offsets, resolve_window
from semantic.schema import DataQuality, Device, Observation, Window

SOURCE = "ga4"

FUNNEL_STEPS = ("view", "cart", "checkout", "purchase")
STEP_FIELD = {"view": "views", "cart": "carts", "checkout": "checkouts", "purchase": "purchases"}
STEP_TRANSITIONS = (("view", "cart"), ("cart", "checkout"), ("checkout", "purchase"))

DIMENSIONS = ("channel", "source_medium", "landing_page", "browser", "os", "country")

# Nombre de la dimensión canónica -> nombre en la Data API
GA4_DIMENSION = {
    "channel": "sessionDefaultChannelGroup",
    "source_medium": "sessionSourceMedium",
    "landing_page": "landingPage",
    "browser": "browser",
    "os": "operatingSystem",
    "country": "country",
}
GA4_EVENT = {"view": "view_item", "cart": "add_to_cart", "checkout": "begin_checkout", "purchase": "purchase"}

CONVERSION_NOTE = (
    "Conversión calculada con los purchases que registra GA4, no con los pedidos "
    "de Shopify. La fuente de verdad de pedidos es Shopify: usa get_orders_summary "
    "o get_conversion_rate si la pregunta implica dinero."
)


class GA4Report(BaseModel):
    """Respuesta cruda del transporte, ya normalizada en forma, no en significado."""

    rows: list[dict] = Field(default_factory=list)
    sampled: bool = False
    sample_size: int | None = None
    property_timezone: str = "Europe/Madrid"
    last_complete_date: date | None = None


@runtime_checkable
class GA4Transport(Protocol):
    property_timezone: str

    async def run_report(
        self, *, start: date, end: date, dimension: str | None, device: Device, limit: int | None
    ) -> GA4Report: ...

    async def ping(self) -> GA4Report: ...


# --------------------------------------------------------------------------- #
# Transporte: fixtures
# --------------------------------------------------------------------------- #
class GA4FixtureTransport:
    """Sirve el dataset de `fixtures/`. No toca la red."""

    def __init__(self, *, property_timezone: str = "Europe/Madrid") -> None:
        from fixtures.dataset import load_dataset

        self._ds = load_dataset()
        self.property_timezone = self._ds.ga4_meta.get("property_timezone", property_timezone)
        self.calls = 0  # lo usan los tests de caché

    async def run_report(
        self, *, start: date, end: date, dimension: str | None, device: Device, limit: int | None
    ) -> GA4Report:
        self.calls += 1
        from fixtures.dataset import allocate

        offsets = _offsets_for_range(start, end, self.property_timezone)
        dev = device.value
        totals = self._ds.ga4_totals(offsets, dev)
        tz = self.property_timezone
        last_complete = _today_in(tz) - timedelta(days=self._ds.ga4_meta.get("consolidation_lag_days", 0))

        if dimension is None:
            rows = [{"dimension_value": None, **totals}]
        else:
            shares = self._ds.dimension_shares(dimension, dev)
            cr_index = self._ds.ga4_shares.get("cr_index", {}).get(dimension, {})
            weighted = {v: s * cr_index.get(v, 1.0) for v, s in shares.items()}
            wsum = sum(weighted.values()) or 1.0
            sessions = allocate(totals["sessions"], shares)
            users = allocate(totals["users"], shares)
            purchases = allocate(totals["purchases"], {v: w / wsum for v, w in weighted.items()})
            carts = allocate(totals["carts"], shares)
            checkouts = allocate(totals["checkouts"], shares)
            rows = [
                {
                    "dimension_value": v,
                    "sessions": sessions[v],
                    "users": users[v],
                    "views": sessions[v],
                    "carts": carts[v],
                    "checkouts": checkouts[v],
                    "purchases": purchases[v],
                }
                for v in shares
            ]
            rows.sort(key=lambda r: (-r["sessions"], r["dimension_value"]))
            if limit:
                rows = rows[:limit]

        return GA4Report(
            rows=rows,
            sampled=bool(self._ds.ga4_meta.get("sampled")),
            sample_size=None,
            property_timezone=tz,
            last_complete_date=last_complete,
        )

    async def ping(self) -> GA4Report:
        self.calls += 1
        tz = self.property_timezone
        return GA4Report(
            rows=[],
            property_timezone=tz,
            last_complete_date=_today_in(tz) - timedelta(days=self._ds.ga4_meta.get("consolidation_lag_days", 0)),
        )


# --------------------------------------------------------------------------- #
# Transporte: API real
# --------------------------------------------------------------------------- #
class GA4ApiTransport:
    """Google Analytics Data API v1beta. Solo lectura, ADC, sin credenciales en repo."""

    CONSOLIDATION_LAG_DAYS = 2  # GA4 no consolida hasta ~48 h

    def __init__(self, property_id: str | None = None, *, property_timezone: str | None = None) -> None:
        self.property_id = property_id or os.environ["CRO_GA4_PROPERTY_ID"]
        self.property_timezone = property_timezone or os.getenv("CRO_GA4_TIMEZONE", "Europe/Madrid")
        self._client = None

    def _get_client(self):
        if self._client is None:  # import perezoso: la dependencia es opcional
            from google.analytics.data_v1beta import BetaAnalyticsDataClient

            self._client = BetaAnalyticsDataClient()
        return self._client

    async def run_report(
        self, *, start: date, end: date, dimension: str | None, device: Device, limit: int | None
    ) -> GA4Report:
        import asyncio

        from google.analytics.data_v1beta.types import (
            DateRange,
            Dimension,
            Filter,
            FilterExpression,
            FilterExpressionList,
            Metric,
            RunReportRequest,
        )
        from google.api_core import exceptions as gexc

        dims = [Dimension(name="eventName")]
        if dimension:
            dims.append(Dimension(name=GA4_DIMENSION[dimension]))

        expressions = [
            FilterExpression(
                filter=Filter(
                    field_name="eventName",
                    in_list_filter=Filter.InListFilter(values=list(GA4_EVENT.values())),
                )
            )
        ]
        if device is not Device.ALL:
            expressions.append(
                FilterExpression(
                    filter=Filter(
                        field_name="deviceCategory",
                        string_filter=Filter.StringFilter(value=device.value),
                    )
                )
            )

        request = RunReportRequest(
            property=f"properties/{self.property_id}",
            date_ranges=[DateRange(start_date=start.isoformat(), end_date=end.isoformat())],
            dimensions=dims,
            metrics=[Metric(name="sessions"), Metric(name="totalUsers"), Metric(name="eventCount")],
            dimension_filter=FilterExpression(and_group=FilterExpressionList(expressions=expressions)),
            limit=limit or 250,
        )

        def _call():
            try:
                return self._get_client().run_report(request)
            except (gexc.TooManyRequests, gexc.ServiceUnavailable) as exc:
                raise TransientSourceError(SOURCE, getattr(exc, "code", 503), str(exc)) from exc

        response = await asyncio.to_thread(_call)
        return self._to_report(response, dimension)

    def _to_report(self, response, dimension: str | None) -> GA4Report:
        event_index = {v: k for k, v in GA4_EVENT.items()}
        agg: dict[str | None, dict] = {}
        for row in response.rows:
            event = row.dimension_values[0].value
            value = row.dimension_values[1].value if dimension else None
            slot = agg.setdefault(
                value,
                {"dimension_value": value, "sessions": 0, "users": 0,
                 "views": 0, "carts": 0, "checkouts": 0, "purchases": 0},
            )
            step = event_index.get(event)
            count = int(row.metric_values[2].value or 0)
            if step:
                slot[STEP_FIELD[step]] += count
            if step == "view":  # sesiones y usuarios se leen una sola vez por fila base
                slot["sessions"] = max(slot["sessions"], int(row.metric_values[0].value or 0))
                slot["users"] = max(slot["users"], int(row.metric_values[1].value or 0))

        sampling = list(getattr(response.metadata, "sampling_metadatas", []) or [])
        sampled = bool(sampling)
        sample_size = int(sampling[0].samples_read_count) if sampled else None
        return GA4Report(
            rows=[strip_pii(r) for r in agg.values()],
            sampled=sampled,
            sample_size=sample_size,
            property_timezone=self.property_timezone,
            last_complete_date=_today_in(self.property_timezone) - timedelta(days=self.CONSOLIDATION_LAG_DAYS),
        )

    async def ping(self) -> GA4Report:
        today = _today_in(self.property_timezone)
        await self.run_report(
            start=today - timedelta(days=2), end=today - timedelta(days=2),
            dimension=None, device=Device.ALL, limit=1,
        )
        return GA4Report(
            property_timezone=self.property_timezone,
            last_complete_date=today - timedelta(days=self.CONSOLIDATION_LAG_DAYS),
        )


# --------------------------------------------------------------------------- #
# Conector
# --------------------------------------------------------------------------- #
class GA4Connector:
    name = SOURCE

    def __init__(self, transport: GA4Transport, *, retry_delay: float = 0.5) -> None:
        self.transport = transport
        self._retry_delay = retry_delay

    def capabilities(self) -> list[Capability]:
        windows = list(Window)
        return [
            Capability(metric="sessions", dimensions=list(DIMENSIONS) + ["device"], windows=windows),
            Capability(metric="users", dimensions=list(DIMENSIONS) + ["device"], windows=windows),
            Capability(metric="funnel_steps", dimensions=["device", "step"], windows=windows),
            Capability(metric="traffic_by_dim", dimensions=list(DIMENSIONS), windows=windows),
        ]

    @property
    def timezone(self) -> str:
        return getattr(self.transport, "property_timezone", "Europe/Madrid")

    def window_dates(self, window: Window) -> tuple[date, date]:
        """Las ventanas se resuelven aquí, en la zona horaria de la propiedad."""
        return resolve_window(window, tz=ZoneInfo(self.timezone))

    async def fetch(self, request: FetchRequest) -> list[Observation]:
        start, end = self.window_dates(request.window)
        dimension = request.dimension if request.metric == "traffic_by_dim" else None
        if dimension and dimension not in DIMENSIONS:
            raise ValueError(f"dimensión no soportada por ga4: {dimension}")

        report = await with_retries(
            lambda: self.transport.run_report(
                start=start, end=end, dimension=dimension, device=request.device, limit=request.limit
            ),
            source=SOURCE,
            base_delay=self._retry_delay,
        )
        quality = self._quality(report, end)
        fetched_at = datetime.now(timezone.utc)
        base = dict(window=request.window, window_start=start, window_end=end,
                    source=SOURCE, quality=quality, fetched_at=fetched_at)
        device_dim = {"device": request.device.value}

        if request.metric == "funnel_steps":
            return self._funnel_observations(report, base, device_dim)
        if request.metric == "traffic_by_dim":
            return self._breakdown_observations(report, base, device_dim, dimension or "channel")
        return self._scalar_observations(report, base, device_dim, request.metric)

    def _quality(self, report: GA4Report, end: date) -> DataQuality:
        """PARTIAL si la ventana llega más allá del último día consolidado."""
        if report.sampled:
            return DataQuality.SAMPLED
        if report.last_complete_date is not None and end > report.last_complete_date:
            return DataQuality.PARTIAL
        return DataQuality.EXACT

    def _partial_note(self, report: GA4Report, end: date) -> str | None:
        if report.last_complete_date is not None and end > report.last_complete_date:
            return (f"Ventana no consolidada: GA4 tiene dato completo hasta {report.last_complete_date}. "
                    "Un dato PARTIAL no soporta una conclusión.")
        return None

    def _scalar_observations(self, report, base, device_dim, metric) -> list[Observation]:
        field = {"sessions": "sessions", "users": "users"}.get(metric)
        if field is None:
            raise ValueError(f"métrica no soportada por ga4: {metric}")
        total = sum(r.get(field, 0) for r in report.rows)
        note = self._partial_note(report, base["window_end"])
        return [Observation(metric=metric, value=float(total), unit="count",
                            dimensions=dict(device_dim), sample_size=total, note=note, **base)]

    def _funnel_observations(self, report, base, device_dim) -> list[Observation]:
        totals = {step: sum(r.get(STEP_FIELD[step], 0) for r in report.rows) for step in FUNNEL_STEPS}
        note = self._partial_note(report, base["window_end"])
        obs: list[Observation] = []
        for step in FUNNEL_STEPS:
            obs.append(Observation(
                metric="funnel_step_volume", value=float(totals[step]), unit="count",
                dimensions={**device_dim, "step": step}, sample_size=totals[step], note=note, **base,
            ))
        for src, dst in STEP_TRANSITIONS:
            denom = totals[src]
            rate = (totals[dst] / denom) if denom else 0.0
            obs.append(Observation(
                metric="funnel_step_rate", value=round(rate, 6), unit="ratio",
                dimensions={**device_dim, "step": f"{src}_to_{dst}"}, sample_size=denom,
                note=note or f"{totals[dst]} de {totals[src]} ({src} -> {dst})", **base,
            ))
        return obs

    def _breakdown_observations(self, report, base, device_dim, dimension) -> list[Observation]:
        partial = self._partial_note(report, base["window_end"])
        obs: list[Observation] = []
        for row in report.rows:
            dims = {**device_dim, dimension: row["dimension_value"]}
            sessions = row.get("sessions", 0)
            purchases = row.get("purchases", 0)
            obs.append(Observation(metric="sessions", value=float(sessions), unit="count",
                                   dimensions=dims, sample_size=sessions, note=partial, **base))
            obs.append(Observation(metric="purchases_ga4", value=float(purchases), unit="count",
                                   dimensions=dims, sample_size=purchases, note=partial, **base))
            rate = (purchases / sessions) if sessions else 0.0
            obs.append(Observation(
                metric="conversion_rate_ga4", value=round(rate, 6), unit="ratio", dimensions=dims,
                sample_size=sessions, note=" ".join(filter(None, [CONVERSION_NOTE, partial])), **base,
            ))
        return obs

    async def health(self) -> HealthStatus:
        started = time.perf_counter()
        try:
            report = await with_retries(self.transport.ping, source=SOURCE, base_delay=self._retry_delay)
        except Exception as exc:  # noqa: BLE001 - health nunca propaga
            return HealthStatus(source=SOURCE, reachable=False, latency_ms=_ms(started), message=str(exc))
        return HealthStatus(
            source=SOURCE,
            reachable=True,
            last_complete_date=report.last_complete_date,
            latency_ms=_ms(started),
            message=f"propiedad en {report.property_timezone}"
            + (" · respuesta muestreada" if report.sampled else ""),
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _today_in(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def _offsets_for_range(start: date, end: date, tz_name: str = "Europe/Madrid") -> list[int]:
    """Traduce un rango de fechas a offsets de día del dataset (1 = ayer)."""
    today = _today_in(tz_name)
    return list(range((today - end).days, (today - start).days + 1))


__all__ = ["GA4ApiTransport", "GA4Connector", "GA4FixtureTransport", "GA4Report", "GA4Transport"]
