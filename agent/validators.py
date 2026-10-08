"""Validación post-hoc de los insights.

El validador numérico es el control más importante del sistema: extrae toda
cifra del texto del modelo y comprueba que procede de un resultado de tool del
mismo run. Una cifra que no esté respaldada es, por definición, inventada.

Además hace cumplir en runtime las dos invariantes del modelo de datos:

  1. Todo `Evidence.tool_call_id` referencia un `ToolCall` del mismo run.
  2. `causal_claim=True` exige evidencia de al menos dos fuentes distintas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from semantic.schema import Insight, Observation, ToolCall

# Tolerancias: 2 decimales para ratios, unidad para enteros,
# céntimo para moneda.
TOL_COUNT = 1.0
TOL_CURRENCY = 0.01
TOL_RATIO = 0.005
TOL_RATIO_PERCENT = 0.1  # la misma cifra escrita como porcentaje, redondeada
TOL_SECONDS = 1.0

# Ruido que no es una cifra de negocio: ventanas, fechas, ids de llamada.
_NOISE_PATTERNS = (
    re.compile(r"\b(?:last|prev)_\d+d\b", re.I),
    re.compile(r"\bsame_period_last_year\b", re.I),
    re.compile(r"\bcall_[\w-]+\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b[A-Za-z]{1,3}-\d{3,}\b"),  # P-001, TCK-00123
    re.compile(r"\bv\d+(?:\.\d+)*\b", re.I),  # v1, v1.2
    re.compile(r"\b[a-z_]+_\d+[a-z]*\b"),  # nombres de dimensión con números
)

_NUMBER_TOKEN = re.compile(r"(?<![\w.,])\d[\d.,]*(?![\w])")


class ValidationIssue(BaseModel):
    code: str
    message: str
    detail: str | None = None


class ValidationReport(BaseModel):
    ok: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    orphan_figures: list[str] = Field(default_factory=list)

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def feedback(self) -> str:
        """Mensaje que se devuelve al modelo en el reintento."""
        lines = ["VALIDACIÓN: tu respuesta no ha pasado el control post-hoc."]
        if self.orphan_figures:
            lines.append(
                "Estas cifras no aparecen en ningún resultado de tool de este run: "
                + ", ".join(self.orphan_figures)
                + ". Elimínalas o sustitúyelas por una cifra que sí esté en un resultado de tool."
            )
        for issue in self.issues:
            if issue.code != "unsupported_figures":
                lines.append(f"{issue.code}: {issue.message}")
        lines.append("Devuelve el JSON corregido. No añadas cifras nuevas.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Extracción e interpretación de cifras
# --------------------------------------------------------------------------- #
def _strip_noise(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def parse_number_candidates(token: str) -> list[float]:
    """Interpretaciones posibles de un token numérico.

    '1.012' puede ser 1012 (separador de miles, español) o 1.012 (decimal,
    inglés). Se aceptan ambas: el objetivo es no acusar de inventar a quien no
    ha inventado, no lucirse con la gramática de los números.
    """
    token = token.strip().rstrip(".,")
    if not token or not any(c.isdigit() for c in token):
        return []
    has_dot, has_comma = "." in token, "," in token
    out: list[float] = []

    def _try(value: str) -> None:
        try:
            out.append(float(value))
        except ValueError:
            pass

    if has_dot and has_comma:
        if token.rfind(".") > token.rfind(","):  # 1,234.56 -> inglés
            _try(token.replace(",", ""))
        else:  # 1.234,56 -> español
            _try(token.replace(".", "").replace(",", "."))
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        head, _, tail = token.rpartition(sep)
        grouped = len(tail) == 3 and sep not in head.replace(sep, "") or _is_grouped(token, sep)
        if grouped:
            _try(token.replace(sep, ""))
        _try(token.replace(sep, ".") if token.count(sep) == 1 else token.replace(sep, ""))
    else:
        _try(token)
    return list(dict.fromkeys(out))


def _is_grouped(token: str, sep: str) -> bool:
    parts = token.split(sep)
    return len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3


def extract_figures(text: str) -> list[tuple[str, list[float]]]:
    cleaned = _strip_noise(text or "")
    figures: list[tuple[str, list[float]]] = []
    for match in _NUMBER_TOKEN.finditer(cleaned):
        token = match.group(0).rstrip(".,")
        candidates = parse_number_candidates(token)
        if candidates:
            figures.append((token, candidates))
    return figures


# --------------------------------------------------------------------------- #
# Cifras soportadas por las observaciones del run
# --------------------------------------------------------------------------- #
@dataclass
class SupportedFigures:
    """Todo lo que el run puede respaldar: valores, y cifras citadas en notas."""

    values: list[tuple[float, str]] = field(default_factory=list)  # (valor, unidad)
    literals: set[float] = field(default_factory=set)  # cifras en notas y dimensiones

    @classmethod
    def from_tool_calls(cls, tool_calls: list[ToolCall]) -> "SupportedFigures":
        supported = cls()
        for call in tool_calls:
            for obs in call.observations:
                supported.add_observation(obs)
            # El texto orientativo que devuelve una tool también es resultado de
            # tool: si dice "n=12", citar 12 no es inventar.
            for _, candidates in extract_figures(call.error or ""):
                supported.literals.update(candidates)
        return supported

    def add_observation(self, obs: Observation) -> None:
        self.values.append((obs.value, obs.unit))
        if obs.sample_size is not None:
            self.values.append((float(obs.sample_size), "count"))
        for text in [obs.note or "", *obs.dimensions.values(), obs.metric]:
            for _, candidates in extract_figures(str(text)):
                self.literals.update(candidates)

    def supports(self, candidates: list[float]) -> bool:
        for candidate in candidates:
            for value, unit in self.values:
                if _matches(candidate, value, unit):
                    return True
            for literal in self.literals:
                if abs(candidate - literal) <= TOL_CURRENCY:
                    return True
        return False


def _matches(candidate: float, value: float, unit: str) -> bool:
    if unit == "count":
        return abs(candidate - value) <= TOL_COUNT
    if unit == "currency_eur":
        return abs(candidate - value) <= TOL_CURRENCY
    if unit == "seconds":
        return abs(candidate - value) <= TOL_SECONDS
    # ratio: admite la forma decimal (0,1215) y la porcentual (12,15)
    return abs(candidate - value) <= TOL_RATIO or abs(candidate - value * 100) <= TOL_RATIO_PERCENT


# --------------------------------------------------------------------------- #
# Validación de un insight
# --------------------------------------------------------------------------- #
def insight_text(insight: Insight) -> str:
    parts = [insight.hypothesis, *insight.assumptions]
    if insight.proposed_action:
        action = insight.proposed_action
        parts += [action.what, action.estimated_impact, action.validation]
    return "\n".join(parts)


def validate_insight(insight: Insight, tool_calls: list[ToolCall]) -> ValidationReport:
    report = ValidationReport()
    by_id = {call.id: call for call in tool_calls}
    supported = SupportedFigures.from_tool_calls(tool_calls)

    # Invariante: toda evidencia referencia una tool_call de ESTE run
    for evidence in insight.evidence:
        call = by_id.get(evidence.tool_call_id)
        if call is None:
            report.issues.append(ValidationIssue(
                code="unknown_tool_call",
                message=f"la evidencia cita tool_call_id '{evidence.tool_call_id}', que no existe en este run",
            ))
            continue
        if not any(
            obs.metric == evidence.metric and _matches(evidence.value, obs.value, obs.unit)
            for obs in call.observations
        ):
            report.issues.append(ValidationIssue(
                code="evidence_value_mismatch",
                message=(f"la evidencia '{evidence.metric}={evidence.value}' no coincide con ninguna "
                         f"observación de {evidence.tool_call_id}"),
            ))

    # Invariante: una causa exige al menos dos fuentes distintas
    if insight.causal_claim and len({e.source for e in insight.evidence}) < 2:
        report.issues.append(ValidationIssue(
            code="causal_single_source",
            message="causal_claim=true con evidencia de una sola fuente",
        ))

    # Validador numérico
    orphans: list[str] = []
    for token, candidates in extract_figures(insight_text(insight)):
        if not supported.supports(candidates):
            orphans.append(token)
    if orphans:
        report.orphan_figures = list(dict.fromkeys(orphans))
        report.issues.append(ValidationIssue(
            code="unsupported_figures",
            message="hay cifras que no proceden de ningún resultado de tool",
            detail=", ".join(report.orphan_figures),
        ))

    report.ok = not report.issues
    return report


def apply_causal_rule(insight: Insight) -> tuple[Insight, bool]:
    """La regla de las dos fuentes, en runtime: degrada la afirmación, no la borra."""
    if insight.causal_claim and len({e.source for e in insight.evidence}) < 2:
        return insight.model_copy(update={
            "causal_claim": False,
            "confidence": "low",
            "assumptions": [*insight.assumptions,
                            "Afirmación causal degradada a correlación: la evidencia procede de una "
                            "sola fuente."],
        }), True
    return insight, False


def degrade(insight: Insight, report: ValidationReport) -> Insight:
    """Segundo fallo: se devuelve el insight sin la afirmación causal."""
    reason = "unsupported_figures" if report.orphan_figures else (
        sorted(report.codes())[0] if report.issues else "validation_failed"
    )
    detail = (" Cifras sin respaldo: " + ", ".join(report.orphan_figures)) if report.orphan_figures else ""
    return insight.model_copy(update={
        "causal_claim": False,
        "confidence": "low",
        "degraded": True,
        "degraded_reason": reason,
        "assumptions": [*insight.assumptions,
                        f"Insight degradado tras fallar la validación post-hoc ({reason}).{detail}"],
        "evidence": [e for e in insight.evidence],
    })


__all__ = [
    "SupportedFigures",
    "ValidationIssue",
    "ValidationReport",
    "apply_causal_rule",
    "degrade",
    "extract_figures",
    "insight_text",
    "parse_number_candidates",
    "validate_insight",
]
