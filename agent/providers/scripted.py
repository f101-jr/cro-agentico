"""Proveedor determinista sin red (`CRO_LLM_PROVIDER=scripted`).

No es un LLM: es un analista de reglas que habla el mismo protocolo que el
proveedor real (mismas tools, mismo formato de salida, mismo contrato de
`Insight`). Existe por dos razones:

  * `evals/run.py` debe ejecutar el set completo **contra fixtures y sin red**.
  * los tests del orquestador, de los validadores y de las personas no pueden
    depender de la variabilidad de un modelo.

Toda cifra que emite procede literalmente de una `Observation` recibida: es la
misma disciplina que se le impone al modelo real, escrita en Python.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field

from agent.providers.base import LLMMessage, LLMResponse, LLMToolCall, LLMUsage
from connectors.windows import COMPARISON_PAIR
from semantic.schema import Device, Window

MODEL_NAME = "scripted-offline"


# --------------------------------------------------------------------------- #
# Formato de cifras (español). El validador numérico entiende ambas formas.
# --------------------------------------------------------------------------- #
def _es(number: str) -> str:
    return number.replace(",", "@").replace(".", ",").replace("@", ".")


def fmt_count(value: float) -> str:
    return _es(f"{int(round(value)):,}")


def fmt_money(value: float) -> str:
    return _es(f"{value:,.2f}") + " €"


def fmt_ratio(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}".replace(".", ",") + "%"


def fmt_signed_ratio(value: float) -> str:
    return ("+" if value >= 0 else "-") + fmt_ratio(abs(value))


def normalize(text: str) -> str:
    lowered = (text or "").casefold()
    return "".join(c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn")


def _device_label(device: Device) -> str:
    return {"mobile": "móvil", "desktop": "escritorio", "tablet": "tablet",
            "all": "todos los dispositivos"}[device.value]


# --------------------------------------------------------------------------- #
# Índice de observaciones recibidas
# --------------------------------------------------------------------------- #
@dataclass
class ObservationRef:
    tool_call_id: str
    tool: str
    data: dict

    @property
    def value(self) -> float:
        return float(self.data["value"])

    def evidence(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "metric": self.data["metric"],
            "value": self.data["value"],
            "window": self.data["window"],
            "dimensions": self.data.get("dimensions", {}),
            "source": self.data["source"],
            "quality": self.data["quality"],
        }


@dataclass
class Index:
    refs: list[ObservationRef] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)
    health: list[dict] = field(default_factory=list)
    failed_tools: list[str] = field(default_factory=list)
    called: list[str] = field(default_factory=list)

    def find(self, metric: str, *, window: Window | str | None = None, **dims: str) -> ObservationRef | None:
        matches = self.all(metric, window=window, **dims)
        return matches[0] if matches else None

    def all(self, metric: str, *, window: Window | str | None = None, **dims: str) -> list[ObservationRef]:
        out: list[ObservationRef] = []
        for ref in self.refs:
            if ref.data["metric"] != metric:
                continue
            if window is not None and ref.data["window"] != str(window):
                continue
            if all(ref.data.get("dimensions", {}).get(k) == v for k, v in dims.items()):
                out.append(ref)
        return out

    def total(self, metric: str, *, window: Window | str | None = None,
              device: str | None = None) -> ObservationRef | None:
        """El total de una métrica: la fila SIN desglose.

        `find()` casa por subconjunto de dimensiones, así que con device='all'
        también casaría la primera fila de un desglose (p. ej. el AOV del bucket
        >75€) y el relato compararía peras con manzanas. Aquí se exige que la
        observación no traiga ninguna clave de dimensión más allá de `device`.
        """
        dims = {"device": device} if device else {}
        for ref in self.all(metric, window=window, **dims):
            if set(ref.data.get("dimensions", {})) <= {"device"}:
                return ref
        return None

    def sources(self, refs: list[ObservationRef]) -> set[str]:
        return {r.data["source"] for r in refs}


def build_index(messages: list[LLMMessage]) -> Index:
    index = Index()
    for message in messages:
        for block in message.content:
            if block.get("type") == "tool_use":
                index.called.append(block.get("name", ""))
            if block.get("type") != "tool_result":
                continue
            try:
                payload = json.loads(block.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            call_id = payload.get("tool_call_id", block.get("tool_use_id", ""))
            tool = payload.get("tool", "")
            if payload.get("guidance"):
                index.guidance.append(payload["guidance"])
            if not payload.get("ok", True):
                index.failed_tools.append(tool)
            index.health.extend(payload.get("health", []))
            for obs in payload.get("observations", []):
                index.refs.append(ObservationRef(call_id, tool, obs))
    return index


# --------------------------------------------------------------------------- #
# Clasificación de la pregunta
# --------------------------------------------------------------------------- #
UNSUPPORTED_TERMS = {
    "margen": "el coste de producto (COGS)",
    "margenes": "el coste de producto (COGS)",
    "coste": "el coste de producto (COGS)",
    "costes": "el coste de producto (COGS)",
    "cogs": "el coste de producto (COGS)",
    "rentabilidad": "el coste de producto (COGS)",
    "rentable": "el coste de producto (COGS)",
    "beneficio": "el coste de producto (COGS)",
    "cac": "el gasto de medios",
    "roas": "el gasto de medios",
    "ltv": "el histórico por cliente, que se descarta en el conector por política de PII",
    "stock": "el inventario",
    "inventario": "el inventario",
    "nps": "las encuestas de satisfacción",
    "competencia": "datos de mercado externos",
    "competidor": "datos de mercado externos",
    "competidores": "datos de mercado externos",
}

CAUSAL_TERMS = ("por que", "porque", "a que se debe", "causa", "explica", "razon", "motivo",
                "que ha pasado", "que esta pasando", "que pasa")


@dataclass
class Plan:
    kind: str
    device: Device
    window: Window
    compare_window: Window
    missing: str | None = None
    dimension: str = "none"


def _detect_device(q: str) -> Device:
    if any(t in q for t in ("movil", "mobile", "telefono", "smartphone")):
        return Device.MOBILE
    if any(t in q for t in ("escritorio", "desktop", "ordenador")):
        return Device.DESKTOP
    if "tablet" in q:
        return Device.TABLET
    return Device.ALL


def _detect_window(q: str, default: Window) -> Window:
    if "ano pasado" in q or "mismo periodo del ano" in q or "interanual" in q:
        return Window.SAME_PERIOD_LAST_YEAR
    if "30 dias" in q or "ultimo mes" in q or "este mes" in q or "treinta dias" in q:
        return Window.LAST_30D
    if "14 dias" in q or "dos semanas" in q or "quince dias" in q:
        return Window.LAST_14D
    if "7 dias" in q or "ultima semana" in q or "esta semana" in q or "siete dias" in q:
        return Window.LAST_7D
    return default


def _detect_dimension(q: str) -> str:
    if any(t in q for t in ("cesta", "importe", "bucket", "rango de precio")):
        return "aov_bucket"
    if any(t in q for t in ("canal", "canales")):
        return "channel"
    if any(t in q for t in ("dispositivo", "device")):
        return "device"
    if any(t in q for t in ("categoria", "tipo de producto")):
        return "product_type"
    return "none"


def classify(question: str, default_window: Window) -> Plan:
    q = normalize(question)
    device = _detect_device(q)
    window = _detect_window(q, default_window)
    compare = COMPARISON_PAIR.get(window, Window.PREV_14D)
    if window is Window.SAME_PERIOD_LAST_YEAR:
        # "frente al año pasado" se lee al derecho: la ventana viva contra la de
        # hace un año, no al revés.
        window, compare = Window.LAST_14D, Window.SAME_PERIOD_LAST_YEAR
    dimension = _detect_dimension(q)

    def plan(kind: str, missing: str | None = None) -> Plan:
        return Plan(kind, device, window, compare, missing=missing, dimension=dimension)

    for term, missing in UNSUPPORTED_TERMS.items():
        if term in q.split() or term in q:
            return plan("unsupported", missing)

    if any(t in q for t in ("salud de", "fiable", "fiabilidad", "consolidad", "calidad del dato", "health")):
        return plan("health")
    if any(t in q for t in ("ticket de soporte", "tickets", "soporte", "atencion al cliente",
                            "queja", "reclamacion")):
        return plan("support")
    if any(t in q for t in ("producto", "productos", "sku", "articulo")):
        return plan("product")
    if any(t in q for t in ("canal", "canales", "trafico", "landing", "pais", "navegador", "campana")):
        return plan("traffic")
    if any(t in q for t in ("funnel", "embudo", "checkout", "carrito", "paso")):
        return plan("funnel")
    if any(t in q for t in CAUSAL_TERMS):
        return plan("causal_conversion")
    if any(t in q for t in ("conversion", "convierte", "cr ")):
        return plan("conversion")
    if any(t in q for t in ("sesion", "sesiones", "visitas", "usuarios")):
        return plan("sessions")
    if any(t in q for t in ("pedido", "pedidos", "ingreso", "ingresos", "ventas", "aov",
                            "ticket medio", "cesta", "factura")):
        return plan("orders")
    return plan("overview")


# --------------------------------------------------------------------------- #
# Planes de llamada
# --------------------------------------------------------------------------- #
def _calls(plan: Plan, round_no: int) -> list[tuple[str, dict]]:
    w, pw, dev = plan.window.value, plan.compare_window.value, plan.device.value
    if plan.kind in ("causal_conversion", "conversion", "overview"):
        if round_no == 0:
            return [
                ("get_conversion_rate", {"window": w, "device": dev}),
                ("compare_windows", {"metric": "conversion_rate", "window_a": w, "window_b": pw,
                                     "device": dev}),
            ]
        if round_no == 1 and plan.kind != "conversion":
            return [
                ("get_funnel_metrics", {"window": w, "device": dev}),
                ("get_funnel_metrics", {"window": pw, "device": dev}),
                ("get_orders_summary", {"window": w, "dimension": "aov_bucket"}),
                ("get_orders_summary", {"window": pw, "dimension": "aov_bucket"}),
                ("compare_windows", {"metric": "aov", "window_a": w, "window_b": pw, "device": dev}),
                ("compare_windows", {"metric": "orders", "window_a": w, "window_b": pw, "device": dev}),
                ("get_support_signal", {"window": w, "topic": "shipping"}),
                ("get_support_signal", {"window": pw, "topic": "shipping"}),
            ]
        return []
    if plan.kind == "sessions":
        return [("get_conversion_rate", {"window": w, "device": dev})] if round_no == 0 else []
    if plan.kind == "orders":
        if round_no == 0:
            calls = [
                ("get_orders_summary", {"window": w, "dimension": plan.dimension}),
                ("compare_windows", {"metric": "revenue", "window_a": w, "window_b": pw, "device": dev}),
            ]
            if plan.dimension != "none":
                # el titular sale del total, no de la primera fila del desglose
                calls.append(("get_orders_summary", {"window": w, "dimension": "none"}))
                calls.append(("get_orders_summary", {"window": pw, "dimension": plan.dimension}))
            return calls
        return []
    if plan.kind == "funnel":
        if round_no == 0:
            return [
                ("get_funnel_metrics", {"window": w, "device": dev}),
                ("get_funnel_metrics", {"window": pw, "device": dev}),
            ]
        return []
    if plan.kind == "traffic":
        if round_no == 0:
            return [
                ("get_traffic_breakdown", {"window": w, "dimension": _traffic_dimension(plan),
                                           "device": dev, "limit": 10}),
                ("get_conversion_rate", {"window": w, "device": dev}),
            ]
        return []
    if plan.kind == "support":
        if round_no == 0:
            return [
                ("get_support_signal", {"window": w, "topic": "all"}),
                ("get_support_signal", {"window": pw, "topic": "all"}),
            ]
        return []
    if plan.kind == "product":
        if round_no == 0:
            return [
                ("get_product_performance", {"window": w, "metric": "revenue", "limit": 10}),
                ("get_orders_summary", {"window": w, "dimension": "none"}),
            ]
        return []
    if plan.kind == "health":
        # La salud de las fuentes se responde con una observación real al lado:
        # un insight sin evidencia no es un insight.
        if round_no == 0:
            return [("get_data_health", {}),
                    ("get_orders_summary", {"window": w, "dimension": "none"})]
        return []
    if plan.kind == "unsupported":
        if round_no == 0:
            return [
                ("get_orders_summary", {"window": w, "dimension": "none"}),
                ("get_product_performance", {"window": w, "metric": "revenue", "limit": 5}),
            ]
        return []
    return []


def _traffic_dimension(plan: Plan) -> str:
    return {"channel": "channel", "none": "channel"}.get(plan.dimension, "channel")


def _fallback_calls(plan: Plan, index: Index) -> list[tuple[str, dict]]:
    """Cuando una tool devuelve 'muestra insuficiente', se sigue su consejo:
    ampliar a todos los dispositivos en lugar de responder con nada."""
    if index.refs or not index.guidance:
        return []
    if plan.device is Device.ALL:
        return [("get_orders_summary", {"window": plan.window.value, "dimension": "none"})]
    return [
        ("get_conversion_rate", {"window": plan.window.value, "device": "all"}),
        ("get_orders_summary", {"window": plan.window.value, "dimension": "none"}),
    ]


# --------------------------------------------------------------------------- #
# Proveedor
# --------------------------------------------------------------------------- #
class ScriptedProvider:
    model = MODEL_NAME

    def __init__(self, *, default_window: Window = Window.LAST_14D) -> None:
        self.default_window = default_window

    async def complete(self, *, system: str, messages: list[LLMMessage], tools: list[dict],
                       max_tokens: int = 4096) -> LLMResponse:
        question = _question_from(messages)
        plan = classify(question, self.default_window)
        index = build_index(messages)
        rounds = sum(1 for m in messages if m.role == "assistant")
        conservative = "VALIDACIÓN" in _last_user_text(messages)

        pending = [] if conservative else _calls(plan, rounds)
        if not pending and not conservative and rounds >= 1:
            pending = _fallback_calls(plan, index)
        if pending:
            calls = [
                LLMToolCall(id=f"call_{rounds}_{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(pending)
            ]
            return LLMResponse(text="", tool_calls=calls, model=self.model,
                               stop_reason="tool_use", usage=_usage(messages, ""))

        payload = self._synthesize(plan, index, conservative)
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        return LLMResponse(text=text, model=self.model, stop_reason="end_turn",
                           usage=_usage(messages, text))

    def cost_eur(self, usage: LLMUsage) -> float:
        return 0.0

    # ------------------------------------------------------------ síntesis --
    def _synthesize(self, plan: Plan, index: Index, conservative: bool) -> dict:
        fell_back = bool(index.guidance) and plan.device is not Device.ALL and not index.all(
            "conversion_rate", device=plan.device.value
        )
        effective = Plan(plan.kind, Device.ALL if fell_back else plan.device, plan.window,
                         plan.compare_window, plan.missing, plan.dimension)
        builder = {
            "causal_conversion": self._causal_conversion,
            "conversion": self._simple_conversion,
            "overview": self._causal_conversion,
            "sessions": self._sessions,
            "orders": self._orders,
            "funnel": self._funnel,
            "traffic": self._traffic,
            "support": self._support,
            "product": self._product,
            "health": self._health,
            "unsupported": self._unsupported,
        }[plan.kind]
        insights = builder(effective, index)
        if not insights:
            insights = [self._no_data(plan, index)]
        if fell_back:
            note = (f"Sin muestra suficiente para {_device_label(plan.device)} en "
                    f"{plan.window.value}: respondo con todos los dispositivos. "
                    + " ".join(index.guidance[:1]))
            for insight in insights:
                insight.setdefault("assumptions", []).append(note)
        if conservative:
            for insight in insights:
                insight["causal_claim"] = False
                insight["confidence"] = "low"
                insight.setdefault("assumptions", []).append(
                    "Respuesta reescrita tras el aviso del validador: solo se citan cifras "
                    "devueltas por tools."
                )
        return {"insights": insights}

    # -------------------------------------------------------------- bloques --
    def _causal_conversion(self, plan: Plan, index: Index) -> list[dict]:
        dev = plan.device.value
        cr_now = index.find("conversion_rate", window=plan.window, device=dev)
        if cr_now is None:
            return []
        cr_prev = index.find("conversion_rate", window=plan.compare_window, device=dev)
        cr_delta = index.find("conversion_rate_delta_rel")
        outside = index.find("conversion_rate_outside_typical_variation")
        step_now = index.find("funnel_step_rate", window=plan.window, step="checkout_to_purchase",
                              device=dev)
        step_prev = index.find("funnel_step_rate", window=plan.compare_window,
                               step="checkout_to_purchase", device=dev)
        aov_now = index.total("aov", window=plan.window, device=dev)
        aov_prev = index.total("aov", window=plan.compare_window, device=dev)
        aov_delta = index.find("aov_delta_rel")
        orders_delta = index.find("orders_delta_abs")
        small_now = index.find("orders", window=plan.window, aov_bucket="<50€")
        small_prev = index.find("orders", window=plan.compare_window, aov_bucket="<50€")
        big_now = index.find("orders", window=plan.window, aov_bucket=">75€")
        big_prev = index.find("orders", window=plan.compare_window, aov_bucket=">75€")
        ship_now = index.find("support_tickets", window=plan.window, topic="shipping")
        ship_prev = index.find("support_tickets", window=plan.compare_window, topic="shipping")

        used = [r for r in (cr_now, cr_prev, cr_delta, step_now, step_prev, aov_now, aov_prev,
                            aov_delta, orders_delta, small_now, small_prev, big_now, big_prev,
                            ship_now, ship_prev) if r is not None]
        sources = index.sources(used)
        moved = outside is not None and outside.value >= 1

        head = (f"La conversión de {_device_label(plan.device)} en {plan.window.value} es "
                f"{fmt_ratio(cr_now.value, 4)}")
        if cr_prev is not None and cr_delta is not None:
            head += (f", frente a {fmt_ratio(cr_prev.value, 4)} en {plan.compare_window.value}: "
                     f"{fmt_signed_ratio(cr_delta.value)} relativo")
        if outside is not None:
            head += (" (fuera de la variación típica del periodo, contraste calculado en Python)"
                     if moved else
                     " (dentro de la variación típica del periodo, contraste calculado en Python)")
        parts = [head + "."]

        if outside is not None and not moved:
            parts.append("No hay aquí una caída que explicar: la diferencia no se distingue del "
                         "ruido del periodo, así que no le busco causa.")
        if step_now is not None and step_prev is not None:
            parts.append(f"El paso checkout → compra pasa de {fmt_ratio(step_prev.value)} a "
                         f"{fmt_ratio(step_now.value)}; los pasos anteriores del funnel apenas "
                         f"se mueven.")
        if aov_now is not None and aov_prev is not None:
            parts.append(f"En el mismo periodo el AOV pasa de {fmt_money(aov_prev.value)} a "
                         f"{fmt_money(aov_now.value)}"
                         + (f" ({fmt_signed_ratio(aov_delta.value)})" if aov_delta else "") + ".")
        if None not in (small_now, small_prev, big_now, big_prev):
            parts.append(f"Los pedidos de cesta pequeña (<50€) pasan de {fmt_count(small_prev.value)} "
                         f"a {fmt_count(small_now.value)} y los de más de 75€ de "
                         f"{fmt_count(big_prev.value)} a {fmt_count(big_now.value)}: cambia la "
                         f"composición de la cesta, no el volumen de demanda.")
        if ship_now is not None and ship_prev is not None:
            parts.append(f"Soporte lo corrobora desde otra fuente: los tickets de envío pasan de "
                         f"{fmt_count(ship_prev.value)} a {fmt_count(ship_now.value)}.")

        causal = bool(moved and len(sources) >= 2 and step_now is not None and ship_now is not None)
        if causal:
            parts.append("Las tres fuentes apuntan a lo mismo: algo encarece el envío o eleva el "
                         "mínimo de compra para las cestas pequeñas, que abandonan en el último paso.")
        elif moved:
            parts.append("Con la evidencia recogida esto es una correlación, no una causa: falta "
                         "contrastarlo con una segunda fuente.")

        action = None
        if moved and causal:
            impacto = ("Recuperar parte de los pedidos de cesta pequeña perdidos, que han pasado de "
                       f"{fmt_count(small_prev.value)} a {fmt_count(small_now.value)}."
                       if small_now is not None and small_prev is not None
                       else "Recuperar parte de los pedidos de cesta pequeña perdidos.")
            action = {
                "what": ("Revisar el umbral de envío gratuito y el coste de envío que ve el usuario "
                         "en checkout para cestas pequeñas; si el umbral ha subido, escalonarlo o "
                         "acompañarlo de un aviso antes del último paso."),
                "effort": "low",
                "estimated_impact": impacto,
                "validation": ("Test A/B de dos semanas sobre el umbral en móvil, midiendo a la vez "
                               "la tasa checkout → compra y el AOV: si la tasa sube y el AOV no se "
                               "desploma, el cambio compensa."),
            }

        assumptions = ["El desglose por cesta es de todos los dispositivos: la tool de pedidos no "
                       "filtra por device."] if small_now is not None else []
        if causal:
            assumptions.append("No hay dato de coste de envío ni de configuración de la tienda: la "
                               "causa se infiere del patrón, no se lee de una fuente.")
        return [{
            "hypothesis": " ".join(parts),
            "causal_claim": causal,
            "confidence": "high" if (causal or not moved) else "medium",
            "evidence": [r.evidence() for r in used],
            "proposed_action": action,
            "assumptions": assumptions,
            "contradictions": _contradictions(index),
        }]

    def _simple_conversion(self, plan: Plan, index: Index) -> list[dict]:
        dev = plan.device.value
        cr = index.find("conversion_rate", window=plan.window, device=dev)
        if cr is None:
            return []
        prev = index.find("conversion_rate", window=plan.compare_window, device=dev)
        delta = index.find("conversion_rate_delta_rel")
        outside = index.find("conversion_rate_outside_typical_variation")
        used = [r for r in (cr, prev, delta, outside) if r]
        text = (f"La conversión de {_device_label(plan.device)} en {plan.window.value} es "
                f"{fmt_ratio(cr.value, 4)}.")
        if prev is not None and delta is not None:
            text += (f" En {plan.compare_window.value} fue {fmt_ratio(prev.value, 4)}: "
                     f"{fmt_signed_ratio(delta.value)} relativo, "
                     + ("fuera" if outside and outside.value >= 1 else "dentro")
                     + " de la variación típica del periodo.")
        text += (" Es una métrica derivada: pedidos de Shopify sobre sesiones de GA4, así que no "
                 "coincide con la conversión que reporta GA4.")
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "high",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": [], "contradictions": _contradictions(index),
        }]

    def _sessions(self, plan: Plan, index: Index) -> list[dict]:
        dev = plan.device.value
        sessions = index.find("sessions", window=plan.window, device=dev)
        if sessions is None:
            return []
        orders = index.total("orders", window=plan.window, device=dev)
        used = [r for r in (sessions, orders) if r]
        text = (f"En {plan.window.value} hubo {fmt_count(sessions.value)} sesiones de "
                f"{_device_label(plan.device)} según GA4.")
        if orders is not None:
            text += f" Shopify registra {fmt_count(orders.value)} pedidos en la misma ventana."
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "high",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": [], "contradictions": [],
        }]

    def _orders(self, plan: Plan, index: Index) -> list[dict]:
        def _total(metric: str) -> ObservationRef | None:
            return index.total(metric, window=plan.window) or index.find(metric, window=plan.window)

        orders, revenue, aov = _total("orders"), _total("revenue"), _total("aov")
        if orders is None or revenue is None:
            return []
        delta = index.find("revenue_delta_rel")
        used = [r for r in (orders, revenue, aov, delta) if r]
        text = (f"En {plan.window.value} hay {fmt_count(orders.value)} pedidos y "
                f"{fmt_money(revenue.value)} de ingresos")
        if aov is not None:
            text += f", con un AOV de {fmt_money(aov.value)}"
        text += "."
        if delta is not None:
            text += (f" Frente a {plan.compare_window.value}, los ingresos varían "
                     f"{fmt_signed_ratio(delta.value)}.")

        if plan.dimension != "none":
            rows = [r for r in index.all("orders", window=plan.window)
                    if plan.dimension in r.data.get("dimensions", {})]
            rows.sort(key=lambda r: -r.value)
            if rows:
                trozos = []
                for row in rows[:6]:
                    key = row.data["dimensions"][plan.dimension]
                    before = index.find("orders", window=plan.compare_window, **{plan.dimension: key})
                    used.append(row)
                    if before is not None:
                        used.append(before)
                        trozos.append(f"{key}: {fmt_count(before.value)} → {fmt_count(row.value)}")
                    else:
                        trozos.append(f"{key}: {fmt_count(row.value)}")
                text += (f" Reparto por {plan.dimension} ({plan.compare_window.value} → "
                         f"{plan.window.value}): " + "; ".join(trozos) + ".")
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "high",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": ["Pedidos netos de cancelados y devoluciones, según el filtro del conector."],
            "contradictions": [],
        }]

    def _funnel(self, plan: Plan, index: Index) -> list[dict]:
        dev = plan.device.value
        steps = [("view_to_cart", "vista → carrito"), ("cart_to_checkout", "carrito → checkout"),
                 ("checkout_to_purchase", "checkout → compra")]
        used: list[ObservationRef] = []
        chunks: list[str] = []
        worst: tuple[float, str] | None = None
        for key, label in steps:
            now = index.find("funnel_step_rate", window=plan.window, step=key, device=dev)
            prev = index.find("funnel_step_rate", window=plan.compare_window, step=key, device=dev)
            if now is None:
                continue
            used.append(now)
            if prev is not None:
                used.append(prev)
                chunks.append(f"{label}: {fmt_ratio(prev.value)} → {fmt_ratio(now.value)}")
                drop = now.value - prev.value
                if worst is None or drop < worst[0]:
                    worst = (drop, label)
            else:
                chunks.append(f"{label}: {fmt_ratio(now.value)}")
        if not used:
            return []
        text = (f"Funnel de {_device_label(plan.device)} ({plan.compare_window.value} → "
                f"{plan.window.value}). " + "; ".join(chunks) + ".")
        if worst is not None:
            text += f" El escalón que más se deteriora es {worst[1]}."
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "high",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": [], "contradictions": [],
        }]

    def _traffic(self, plan: Plan, index: Index) -> list[dict]:
        rows = [r for r in index.all("sessions") if "channel" in r.data.get("dimensions", {})]
        if not rows:
            return []
        rows.sort(key=lambda r: -r.value)
        used: list[ObservationRef] = []
        chunks = []
        for ref in rows[:4]:
            channel = ref.data["dimensions"]["channel"]
            used.append(ref)
            cr = index.find("conversion_rate_ga4", channel=channel, window=plan.window)
            if cr is not None:
                used.append(cr)
                chunks.append(f"{channel}: {fmt_count(ref.value)} sesiones y "
                              f"{fmt_ratio(cr.value)} de conversión")
            else:
                chunks.append(f"{channel}: {fmt_count(ref.value)} sesiones")
        text = (f"Desglose por canal en {plan.window.value} para {_device_label(plan.device)}. "
                + "; ".join(chunks) + ". La conversión de este desglose la calcula GA4 con sus "
                "propios purchases; la fuente de verdad de pedidos es Shopify.")
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "medium",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": ["La conversión por canal procede de GA4, que subcuenta pedidos."],
            "contradictions": _contradictions(index),
        }]

    def _support(self, plan: Plan, index: Index) -> list[dict]:
        now = index.all("support_tickets", window=plan.window)
        prev = {r.data["dimensions"]["topic"]: r
                for r in index.all("support_tickets", window=plan.compare_window)}
        rows = [r for r in now if r.data["dimensions"].get("topic") not in (None, "all")]
        if not rows:
            return []
        rows.sort(key=lambda r: -r.value)
        used: list[ObservationRef] = []
        chunks = []
        for ref in rows[:5]:
            topic = ref.data["dimensions"]["topic"]
            used.append(ref)
            before = prev.get(topic)
            if before is not None and before.value:
                used.append(before)
                chunks.append(f"{topic}: {fmt_count(before.value)} → {fmt_count(ref.value)}")
            else:
                chunks.append(f"{topic}: {fmt_count(ref.value)}")
        text = (f"Tickets de soporte por tema ({plan.compare_window.value} → {plan.window.value}): "
                + "; ".join(chunks) + ". El tema se asigna por palabras clave en código, no por modelo.")
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "high",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": [], "contradictions": [],
        }]

    def _product(self, plan: Plan, index: Index) -> list[dict]:
        rows = index.all("product_revenue", window=plan.window) or \
               index.all("product_units", window=plan.window)
        if not rows:
            return []
        rows.sort(key=lambda r: -r.value)
        top = rows[:5]
        chunks = [
            f"{r.data['dimensions'].get('product', '?')}: "
            + (fmt_money(r.value) if r.data["unit"] == "currency_eur"
               else f"{fmt_count(r.value)} unidades")
            for r in top
        ]
        text = (f"Productos por ingresos de línea en {plan.window.value}: " + "; ".join(chunks) +
                ". Son importes de línea de producto: no cuadran con el revenue del pedido porque "
                "envío y descuentos van a nivel de pedido.")
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "medium",
            "evidence": [r.evidence() for r in top], "proposed_action": None,
            "assumptions": ["Sin coste de producto no se puede ordenar por rentabilidad."],
            "contradictions": [],
        }]

    def _health(self, plan: Plan, index: Index) -> list[dict]:
        if not index.health:
            return []
        chunks = [
            f"{h['source']}: {'operativa' if h['reachable'] else 'sin respuesta'}"
            + (f", dato completo hasta {h['last_complete_date']}" if h.get("last_complete_date") else "")
            for h in index.health
        ]
        orders = index.find("orders", window=plan.window)
        used = [orders] if orders is not None else []
        text = "Estado de las fuentes. " + "; ".join(chunks) + "."
        if orders is not None:
            text += (f" Como referencia de que el dato llega, la ventana {plan.window.value} trae "
                     f"{fmt_count(orders.value)} pedidos.")
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "high",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": ["Respuesta de diagnóstico: no interpreta métricas de negocio."],
            "contradictions": [],
        }]

    def _unsupported(self, plan: Plan, index: Index) -> list[dict]:
        orders = index.find("orders", window=plan.window)
        revenue = index.find("revenue", window=plan.window)
        top = sorted(index.all("product_revenue", window=plan.window), key=lambda r: -r.value)[:3]
        used = [r for r in (orders, revenue, *top) if r]
        missing = plan.missing or "el dato necesario"
        text = (f"No puedo responder a esta pregunta con las fuentes conectadas: falta {missing}, "
                f"y ninguna de las tres (GA4, Shopify, soporte) lo expone. No voy a estimarlo.")
        if used:
            trozos = []
            if orders is not None and revenue is not None:
                trozos.append(f"{fmt_count(orders.value)} pedidos y {fmt_money(revenue.value)} "
                              f"de ingresos en {plan.window.value}")
            if top:
                trozos.append("ingresos de línea por producto ("
                              + "; ".join(f"{r.data['dimensions'].get('product', '?')}: "
                                          f"{fmt_money(r.value)}" for r in top) + ")")
            text += " Lo que sí tengo: " + "; ".join(trozos) + "."
        text += " Para responderla haría falta conectar una fuente con esa información."
        return [{
            "hypothesis": text, "causal_claim": False, "confidence": "low",
            "evidence": [r.evidence() for r in used], "proposed_action": None,
            "assumptions": [f"Falta {missing}: la pregunta queda sin responder, no estimada."],
            "contradictions": [], "degraded": True, "degraded_reason": "missing_source_data",
        }]

    def _no_data(self, plan: Plan, index: Index) -> dict:
        reasons = index.guidance or ["Las tools no han devuelto observaciones para esta pregunta."]
        evidence = [index.refs[0].evidence()] if index.refs else []
        return {
            "hypothesis": ("No tengo datos suficientes para responder. " + " ".join(reasons[:3])
                           + " No relleno el hueco con una estimación."),
            "causal_claim": False, "confidence": "low", "evidence": evidence,
            "proposed_action": None,
            "assumptions": ["Faltan datos: " + " ".join(reasons[:3])],
            "contradictions": [], "degraded": True, "degraded_reason": "insufficient_data",
        }


def _contradictions(index: Index) -> list[str]:
    """Detecta fuentes que no cuadran entre sí. No promedia: lo recoge."""
    out: list[str] = []
    derived = index.find("conversion_rate")
    ga4_rows = index.all("conversion_rate_ga4")
    if derived is not None and ga4_rows:
        best = max(ga4_rows, key=lambda r: r.data.get("sample_size") or 0)
        if best.value and abs(best.value - derived.value) / max(derived.value, 1e-9) > 0.15:
            out.append(
                f"GA4 reporta una conversión de {fmt_ratio(best.value)} frente a "
                f"{fmt_ratio(derived.value)} de la métrica derivada con pedidos de Shopify: no se "
                "promedian, la fuente de verdad de pedidos es Shopify."
            )
    return out


def _question_from(messages: list[LLMMessage]) -> str:
    for message in messages:
        if message.role != "user":
            continue
        for block in message.content:
            if block.get("type") == "text":
                return block.get("text", "")
    return ""


def _last_user_text(messages: list[LLMMessage]) -> str:
    for message in reversed(messages):
        if message.role != "user":
            continue
        for block in message.content:
            if block.get("type") == "text":
                return block.get("text", "")
    return ""


def _usage(messages: list[LLMMessage], output: str) -> LLMUsage:
    prompt_chars = sum(len(json.dumps(m.content, ensure_ascii=False)) for m in messages)
    return LLMUsage(prompt_tokens=prompt_chars // 4, completion_tokens=len(output) // 4)


__all__ = ["MODEL_NAME", "ScriptedProvider", "classify", "fmt_count", "fmt_money", "fmt_ratio"]
