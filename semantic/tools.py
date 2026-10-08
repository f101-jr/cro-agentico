"""Las ocho tools expuestas al LLM.

Reglas transversales que este módulo hace cumplir:

  * Toda tool valida sus argumentos con Pydantic **antes** de tocar un conector.
  * Los errores vuelven al modelo como texto orientativo, nunca como excepción.
  * Ninguna tool devuelve más de 25 filas; el truncado se declara en `note`.
  * Cada respuesta se cachea por (tool, argumentos) con TTL por ventana.
  * Ninguna cifra se calcula fuera de Python: el modelo no resta ni divide.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from connectors.base import Connector, FetchRequest, HealthStatus, SourceUnavailableError
from connectors.cache import Cache, NullCache, cache_key, ttl_for_window
from connectors.ga4 import DIMENSIONS as GA4_DIMENSIONS
from connectors.windows import resolve_window
from semantic.schema import DataQuality, Device, Observation, Window
from semantic.truth import (
    COMPARABLE_METRICS,
    DERIVED_METRICS,
    derived_note,
    unsupported_reason,
)

MAX_ROWS = 25  # máximo duro: ninguna tool devuelve más filas
MIN_SAMPLE_FOR_RATE = 30  # por debajo de esto una tasa no significa nada
Z_SIGNIFICANT = 1.96


# --------------------------------------------------------------------------- #
# Contexto y resultado
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    ga4: Connector
    shopify: Connector
    support: Connector
    cache: Cache = field(default_factory=NullCache)
    extra: list[Connector] = field(default_factory=list)
    timezone_name: str = "Europe/Madrid"

    @property
    def connectors(self) -> list[Connector]:
        return [self.ga4, self.shopify, self.support, *self.extra]

    def today(self) -> date:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(self.timezone_name)).date()


class ToolResult(BaseModel):
    tool: str
    ok: bool = True
    observations: list[Observation] = Field(default_factory=list)
    health: list[HealthStatus] = Field(default_factory=list)
    guidance: str | None = None  # texto orientativo para el modelo
    truncated: bool = False

    def for_model(self, tool_call_id: str) -> dict:
        """Lo que ve el modelo. Incluye el id de la llamada: es lo que debe citar."""
        payload: dict[str, Any] = {"tool_call_id": tool_call_id, "tool": self.tool, "ok": self.ok}
        if self.guidance:
            payload["guidance"] = self.guidance
        if self.health:
            payload["health"] = [h.model_dump(mode="json") for h in self.health]
        if self.observations:
            payload["observations"] = [
                o.model_dump(mode="json", exclude={"fetched_at"}) for o in self.observations
            ]
        return payload


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], Awaitable[ToolResult]]

    def json_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


# --------------------------------------------------------------------------- #
# Modelos de argumentos
# --------------------------------------------------------------------------- #
class FunnelArgs(BaseModel):
    window: Window
    device: Device = Device.ALL
    step: Literal["view", "cart", "checkout", "purchase", "all"] = "all"


class TrafficArgs(BaseModel):
    window: Window
    dimension: Literal["channel", "source_medium", "landing_page", "browser", "os", "country"]
    device: Device = Device.ALL
    limit: int = Field(default=10, ge=1, le=MAX_ROWS)


class OrdersArgs(BaseModel):
    window: Window
    dimension: Literal["none", "device", "channel", "product_type", "aov_bucket"] = "none"


class ProductArgs(BaseModel):
    window: Window
    metric: Literal["units", "revenue", "conversion"]
    limit: int = Field(default=10, ge=1, le=MAX_ROWS)


class ConversionArgs(BaseModel):
    window: Window
    device: Device = Device.ALL


class CompareArgs(BaseModel):
    metric: str
    window_a: Window
    window_b: Window
    device: Device = Device.ALL


class SupportArgs(BaseModel):
    window: Window
    topic: Literal["shipping", "payment", "product", "returns", "account", "all"] = "all"


class HealthArgs(BaseModel):
    pass


# --------------------------------------------------------------------------- #
# Descripciones para el modelo
# --------------------------------------------------------------------------- #
DESCRIPTIONS = {
    "get_funnel_metrics": (
        "Escalones del funnel y su tasa de paso. Úsala para localizar EN QUÉ PASO se pierde "
        "el usuario. No la uses para ingresos ni pedidos: usa get_orders_summary."
    ),
    "get_traffic_breakdown": (
        "Desglose de sesiones y su conversión por dimensión. Úsala para saber si un cambio se "
        "concentra en un segmento o es transversal. La conversión de esta tool la calcula GA4; "
        "la fuente de verdad de pedidos es Shopify."
    ),
    "get_orders_summary": (
        "Fuente de verdad para pedidos e ingresos. Úsala siempre que la pregunta implique dinero. "
        "Devuelve orders, revenue y AOV; aov_bucket parte en <50€ / 50-75€ / >75€."
    ),
    "get_product_performance": (
        "Rendimiento por producto: unidades, ingresos de línea o proporción de pedidos que lo "
        "incluyen. No devuelve margen ni coste: esas cifras no existen en las fuentes conectadas."
    ),
    "get_conversion_rate": (
        "Conversión como pedidos de Shopify entre sesiones de GA4. Métrica derivada: la "
        "Observation SIEMPRE incluye la nota de composición, léela antes de comparar con GA4."
    ),
    "compare_windows": (
        "Úsala SIEMPRE para comparar dos periodos. No restes tú los valores: devuelve delta "
        "absoluto, delta relativo y si la diferencia supera la variación típica del periodo."
    ),
    "get_support_signal": (
        "Volumen de tickets de soporte por tema. Úsala para contrastar una hipótesis de "
        "comportamiento con lo que los clientes están diciendo."
    ),
    "get_data_health": (
        "Estado de las fuentes y hasta qué día hay dato consolidado. Consúltala si un dato te "
        "resulta anómalo o si la ventana incluye días recientes. Un dato PARTIAL no soporta una "
        "conclusión."
    ),
}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _truncate(observations: list[Observation], *, keys_per_row: int, limit: int) -> tuple[list[Observation], bool]:
    """Acota a `limit` filas lógicas (cada fila puede traer varias Observation)."""
    max_rows = min(limit, MAX_ROWS)
    seen: list[str] = []
    kept: list[Observation] = []
    for obs in observations:
        row = _row_key(obs)
        if row not in seen:
            if len(seen) >= max_rows:
                continue
            seen.append(row)
        kept.append(obs)
    total_rows = len({_row_key(o) for o in observations})
    truncated = total_rows > max_rows
    if truncated:
        note = f" Resultado truncado a {max_rows} filas de {total_rows}."
        kept = [o.model_copy(update={"note": (o.note or "") + note}) for o in kept]
    return kept, truncated


def _row_key(obs: Observation) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(obs.dimensions.items()))


def _sum_value(observations: list[Observation], metric: str) -> float:
    return sum(o.value for o in observations if o.metric == metric)


def _thin_sample_guidance(device: Device, window: Window, n: int) -> str:
    return (f"Sin datos suficientes para device={device.value} en {window.value} (n={n}). "
            f"Prueba device='all' o una ventana mayor.")


def _z_two_proportions(p_a: float, n_a: int, p_b: float, n_b: int) -> float | None:
    if n_a <= 0 or n_b <= 0:
        return None
    pooled = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n_a + 1 / n_b), 1e-12))
    return (p_a - p_b) / se if se else None


def _significance(unit: str, a: float, b: float, n_a: int | None, n_b: int | None) -> tuple[bool, str]:
    """Toda la estadística ocurre aquí, en Python. El modelo solo lee el veredicto."""
    if unit == "ratio" and n_a and n_b:
        z = _z_two_proportions(a, n_a, b, n_b)
        if z is not None:
            return abs(z) >= Z_SIGNIFICANT, (
                f"Contraste de dos proporciones sobre n={n_a} y n={n_b}: z={z:.2f}; "
                f"se considera fuera de la variación típica si |z| >= {Z_SIGNIFICANT}."
            )
    if unit == "count":
        denom = math.sqrt(max(a + b, 1e-9))
        z = (a - b) / denom
        return abs(z) >= Z_SIGNIFICANT, (
            f"Contraste de conteos (aproximación de Poisson): z={z:.2f}; fuera de la variación "
            f"típica si |z| >= {Z_SIGNIFICANT}."
        )
    rel = abs((a - b) / b) if b else 0.0
    n = min(n_a or 0, n_b or 0)
    significant = rel >= 0.05 and n >= 100
    return significant, (
        f"Sin varianza por pedido en la fuente: se usa un umbral declarado del 5% relativo con "
        f"n>=100 (delta={rel:.1%}, n={n})."
    )


# --------------------------------------------------------------------------- #
# Tool 1 — get_funnel_metrics
# --------------------------------------------------------------------------- #
async def get_funnel_metrics(ctx: ToolContext, args: FunnelArgs) -> ToolResult:
    obs = await ctx.ga4.fetch(FetchRequest(metric="funnel_steps", window=args.window, device=args.device))
    if args.step != "all":
        obs = [
            o for o in obs
            if o.dimensions.get("step") == args.step
            or (o.metric == "funnel_step_rate" and args.step in o.dimensions.get("step", "").split("_to_"))
        ]
        if not obs:
            return ToolResult(tool="get_funnel_metrics", ok=False,
                              guidance=f"No hay datos del escalón '{args.step}' en {args.window.value}.")
    return ToolResult(tool="get_funnel_metrics", observations=obs)


# --------------------------------------------------------------------------- #
# Tool 2 — get_traffic_breakdown
# --------------------------------------------------------------------------- #
async def get_traffic_breakdown(ctx: ToolContext, args: TrafficArgs) -> ToolResult:
    if args.dimension not in GA4_DIMENSIONS:
        return ToolResult(tool="get_traffic_breakdown", ok=False,
                          guidance=f"Dimensión no disponible. Opciones: {', '.join(GA4_DIMENSIONS)}.")
    # El recorte lo hace la capa semántica, no la fuente: solo así la Observation
    # puede declarar que se ha truncado y cuánto se ha dejado fuera.
    obs = await ctx.ga4.fetch(FetchRequest(
        metric="traffic_by_dim", window=args.window, device=args.device,
        dimension=args.dimension, limit=None,
    ))
    obs, truncated = _truncate(obs, keys_per_row=3, limit=args.limit)
    return ToolResult(tool="get_traffic_breakdown", observations=obs, truncated=truncated)


# --------------------------------------------------------------------------- #
# Tool 3 — get_orders_summary
# --------------------------------------------------------------------------- #
async def get_orders_summary(ctx: ToolContext, args: OrdersArgs) -> ToolResult:
    obs = await ctx.shopify.fetch(FetchRequest(
        metric="orders_summary", window=args.window, device=Device.ALL, dimension=args.dimension,
    ))
    obs, truncated = _truncate(obs, keys_per_row=3, limit=MAX_ROWS)
    return ToolResult(tool="get_orders_summary", observations=obs, truncated=truncated)


# --------------------------------------------------------------------------- #
# Tool 4 — get_product_performance
# --------------------------------------------------------------------------- #
async def get_product_performance(ctx: ToolContext, args: ProductArgs) -> ToolResult:
    obs = await ctx.shopify.fetch(FetchRequest(
        metric="product_performance", window=args.window, device=Device.ALL, limit=None,
    ))
    key = {"units": "product_units", "revenue": "product_revenue", "conversion": "product_order_rate"}[args.metric]
    ranking = sorted([o for o in obs if o.metric == key], key=lambda o: -o.value)
    order = [_row_key(o) for o in ranking]
    obs = sorted(obs, key=lambda o: (order.index(_row_key(o)) if _row_key(o) in order else 99, o.metric))
    obs, truncated = _truncate(obs, keys_per_row=3, limit=args.limit)
    guidance = None
    if args.metric == "conversion":
        guidance = ("'conversion' aquí es la proporción de pedidos que incluyen el producto: "
                    "Shopify no ve sesiones por producto.")
    return ToolResult(tool="get_product_performance", observations=obs, truncated=truncated, guidance=guidance)


# --------------------------------------------------------------------------- #
# Tool 5 — get_conversion_rate  (métrica derivada)
# --------------------------------------------------------------------------- #
async def get_conversion_rate(ctx: ToolContext, args: ConversionArgs) -> ToolResult:
    sessions_obs, orders_obs = await asyncio.gather(
        ctx.ga4.fetch(FetchRequest(metric="sessions", window=args.window, device=args.device)),
        ctx.shopify.fetch(FetchRequest(metric="orders_summary", window=args.window, device=args.device)),
    )
    sessions = _sum_value(sessions_obs, "sessions")
    orders = _sum_value(orders_obs, "orders")
    if orders < MIN_SAMPLE_FOR_RATE:
        return ToolResult(tool="get_conversion_rate", ok=False,
                          guidance=_thin_sample_guidance(args.device, args.window, int(orders)))
    if sessions <= 0:
        return ToolResult(tool="get_conversion_rate", ok=False,
                          guidance=f"GA4 no devuelve sesiones para device={args.device.value} en "
                                   f"{args.window.value}; sin denominador no hay conversión.")

    spec = DERIVED_METRICS["conversion_rate"]
    start, end = resolve_window(args.window)
    quality = DataQuality.PARTIAL if any(o.quality is DataQuality.PARTIAL for o in sessions_obs) else spec["quality"]
    derived = Observation(
        metric="conversion_rate", value=round(orders / sessions, 6), unit="ratio",
        window=args.window, window_start=start, window_end=end,
        dimensions={"device": args.device.value}, source="derived", quality=quality,
        sample_size=int(sessions), note=f"{spec['note']} Fórmula: {spec['formula']}.",
        fetched_at=datetime.now(timezone.utc),
    )
    components = [o for o in sessions_obs if o.metric == "sessions"] + \
                 [o for o in orders_obs if o.metric in ("orders", "revenue", "aov")]
    return ToolResult(tool="get_conversion_rate", observations=[derived, *components])


# --------------------------------------------------------------------------- #
# Tool 6 — compare_windows  (TODO EL CÁLCULO EN PYTHON)
# --------------------------------------------------------------------------- #
async def _metric_observation(ctx: ToolContext, metric: str, window: Window, device: Device) -> Observation | None:
    if metric in ("sessions", "users"):
        obs = await ctx.ga4.fetch(FetchRequest(metric=metric, window=window, device=device))
        return next((o for o in obs if o.metric == metric), None)
    if metric in ("orders", "revenue", "aov"):
        obs = await ctx.shopify.fetch(FetchRequest(metric="orders_summary", window=window, device=device))
        return next((o for o in obs if o.metric == metric), None)
    if metric == "refunds":
        obs = await ctx.shopify.fetch(FetchRequest(metric="refunds", window=window, device=device))
        return next((o for o in obs if o.metric == "refunds"), None)
    if metric == "support_tickets":
        obs = await ctx.support.fetch(FetchRequest(metric="support_tickets", window=window))
        return next((o for o in obs if o.dimensions.get("topic") == "all"), None)
    if metric == "conversion_rate":
        result = await get_conversion_rate(ctx, ConversionArgs(window=window, device=device))
        return next((o for o in result.observations if o.metric == "conversion_rate"), None)
    return None


async def compare_windows(ctx: ToolContext, args: CompareArgs) -> ToolResult:
    reason = unsupported_reason(args.metric)
    if reason:
        return ToolResult(tool="compare_windows", ok=False,
                          guidance=f"No puedo comparar '{args.metric}': {reason}. No hay forma de "
                                   f"obtener esa cifra con las fuentes conectadas.")
    if args.metric not in COMPARABLE_METRICS:
        return ToolResult(tool="compare_windows", ok=False,
                          guidance=f"Métrica no comparable. Opciones: {', '.join(COMPARABLE_METRICS)}.")
    if args.window_a == args.window_b:
        return ToolResult(tool="compare_windows", ok=False,
                          guidance="window_a y window_b son la misma ventana: no hay nada que comparar.")

    obs_a, obs_b = await asyncio.gather(
        _metric_observation(ctx, args.metric, args.window_a, args.device),
        _metric_observation(ctx, args.metric, args.window_b, args.device),
    )
    if obs_a is None or obs_b is None:
        missing = args.window_a.value if obs_a is None else args.window_b.value
        return ToolResult(tool="compare_windows", ok=False,
                          guidance=f"Sin datos de '{args.metric}' para {missing} y device="
                                   f"{args.device.value}. Prueba device='all' o una ventana mayor.")

    a, b = obs_a.value, obs_b.value
    delta_abs = round(a - b, 6)
    delta_rel = round((a - b) / b, 6) if b else 0.0
    significant, criterion = _significance(obs_a.unit, a, b, obs_a.sample_size, obs_b.sample_size)
    dims = {"device": args.device.value, "window_a": args.window_a.value, "window_b": args.window_b.value}
    verdict = "supera la variación típica" if significant else "dentro de la variación típica"
    note = (f"{args.metric}: {a} ({args.window_a.value}) frente a {b} ({args.window_b.value}). "
            f"Diferencia calculada en Python; {verdict}. {criterion}")
    start_a, end_a = obs_a.window_start, obs_a.window_end
    common = dict(window=args.window_a, window_start=start_a, window_end=end_a,
                  source=obs_a.source, quality=obs_a.quality, fetched_at=datetime.now(timezone.utc))

    return ToolResult(tool="compare_windows", observations=[
        obs_a, obs_b,
        Observation(metric=f"{args.metric}_delta_abs", value=delta_abs, unit=obs_a.unit,
                    dimensions=dims, sample_size=obs_a.sample_size, note=note, **common),
        Observation(metric=f"{args.metric}_delta_rel", value=delta_rel, unit="ratio",
                    dimensions=dims, sample_size=obs_a.sample_size,
                    note=f"Variación relativa {delta_rel:+.2%} respecto a {args.window_b.value}. {note}",
                    **common),
        Observation(metric=f"{args.metric}_outside_typical_variation", value=float(significant), unit="count",
                    dimensions=dims, sample_size=obs_a.sample_size,
                    note=f"1 = fuera de la variación típica, 0 = dentro. {criterion}", **common),
    ])


# --------------------------------------------------------------------------- #
# Tool 7 — get_support_signal
# --------------------------------------------------------------------------- #
async def get_support_signal(ctx: ToolContext, args: SupportArgs) -> ToolResult:
    obs = await ctx.support.fetch(FetchRequest(
        metric="support_tickets", window=args.window, filters={"topic": args.topic},
    ))
    obs, truncated = _truncate(obs, keys_per_row=1, limit=MAX_ROWS)
    return ToolResult(tool="get_support_signal", observations=obs, truncated=truncated)


# --------------------------------------------------------------------------- #
# Tool 8 — get_data_health
# --------------------------------------------------------------------------- #
async def get_data_health(ctx: ToolContext, args: HealthArgs) -> ToolResult:
    statuses = await asyncio.gather(*(c.health() for c in ctx.connectors), return_exceptions=True)
    health: list[HealthStatus] = []
    for connector, status in zip(ctx.connectors, statuses):
        if isinstance(status, BaseException):
            health.append(HealthStatus(source=getattr(connector, "name", "?"), reachable=False,
                                       message=str(status)))
        else:
            health.append(status)
    stale = [h.source for h in health if not h.reachable]
    guidance = (f"Fuentes que no responden: {', '.join(stale)}. Cualquier conclusión que dependa de "
                f"ellas queda sin soporte.") if stale else None
    return ToolResult(tool="get_data_health", health=health, guidance=guidance)


# --------------------------------------------------------------------------- #
# Registro y ejecución
# --------------------------------------------------------------------------- #
def build_registry() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec("get_funnel_metrics", DESCRIPTIONS["get_funnel_metrics"], FunnelArgs, get_funnel_metrics),
        ToolSpec("get_traffic_breakdown", DESCRIPTIONS["get_traffic_breakdown"], TrafficArgs, get_traffic_breakdown),
        ToolSpec("get_orders_summary", DESCRIPTIONS["get_orders_summary"], OrdersArgs, get_orders_summary),
        ToolSpec("get_product_performance", DESCRIPTIONS["get_product_performance"], ProductArgs,
                 get_product_performance),
        ToolSpec("get_conversion_rate", DESCRIPTIONS["get_conversion_rate"], ConversionArgs, get_conversion_rate),
        ToolSpec("compare_windows", DESCRIPTIONS["compare_windows"], CompareArgs, compare_windows),
        ToolSpec("get_support_signal", DESCRIPTIONS["get_support_signal"], SupportArgs, get_support_signal),
        ToolSpec("get_data_health", DESCRIPTIONS["get_data_health"], HealthArgs, get_data_health),
    ]
    return {spec.name: spec for spec in specs}


REGISTRY = build_registry()
TOOL_NAMES = tuple(REGISTRY)


def tool_schemas() -> list[dict]:
    return [spec.json_schema() for spec in REGISTRY.values()]


def _validation_guidance(name: str, exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(argumento)"
        expected = err.get("ctx", {}).get("expected")
        parts.append(f"{loc}: {err['msg']}" + (f" (valores válidos: {expected})" if expected else ""))
    return (f"Argumentos inválidos para {name}: " + "; ".join(parts) +
            ". Corrige la llamada y vuelve a intentarlo.")


async def execute_tool(ctx: ToolContext, name: str, arguments: dict, *, use_cache: bool = True) -> ToolResult:
    """Ejecuta una tool. NUNCA lanza: todo error vuelve como texto orientativo."""
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolResult(tool=name, ok=False,
                          guidance=f"No existe la tool '{name}'. Disponibles: {', '.join(TOOL_NAMES)}.")
    try:
        args = spec.args_model.model_validate(arguments or {})
    except ValidationError as exc:
        return ToolResult(tool=name, ok=False, guidance=_validation_guidance(name, exc))

    key = cache_key(name, args.model_dump(mode="json"))
    if use_cache:
        cached = await ctx.cache.get(key)
        if cached is not None:
            return ToolResult.model_validate(cached)

    try:
        result = await spec.handler(ctx, args)
    except SourceUnavailableError as exc:
        return ToolResult(tool=name, ok=False,
                          guidance=f"La fuente '{exc.source}' no responde: {exc.message}. No uses esta "
                                   f"métrica para concluir; comprueba get_data_health.")
    except ValueError as exc:
        return ToolResult(tool=name, ok=False, guidance=f"{name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - el modelo nunca ve una excepción
        return ToolResult(tool=name, ok=False,
                          guidance=f"{name} ha fallado ({type(exc).__name__}): {exc}. Prueba otra "
                                   f"ventana o dimensión.")

    if use_cache and result.ok:
        ends = [o.window_end for o in result.observations]
        ttl = ttl_for_window(max(ends), ctx.today()) if ends else 60.0
        await ctx.cache.set(key, result.model_dump(mode="json"), ttl)
    return result


__all__ = [
    "DESCRIPTIONS",
    "MAX_ROWS",
    "MIN_SAMPLE_FOR_RATE",
    "REGISTRY",
    "TOOL_NAMES",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "execute_tool",
    "tool_schemas",
]
