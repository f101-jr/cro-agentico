"""Personas.

Una persona **solo** modifica la síntesis final y la agregación de salida. No
cambia qué tools existen, ni qué evidencia se recoge, ni el pipeline: la
proyección se aplica sobre los insights canónicos ya validados, de modo que el
mismo `run_id` se puede renderizar con las tres y las cifras son, por
construcción, idénticas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from semantic.schema import Insight, ProposedAction

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "personas"
PERSONAS = ("ceo", "ecommerce_manager", "analyst")
DEFAULT_PERSONA = "analyst"
SUMMARY_EVIDENCE_LIMIT = 4


class UnknownPersona(ValueError):
    pass


@dataclass(frozen=True)
class Persona:
    name: str
    version: str
    max_insights: int | None
    proposed_action: str  # optional | required
    evidence: str  # summary | full
    assumptions: str  # visible | hidden
    confidence: str  # label | label_and_sample
    horizon: str
    prompt: str


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    _, _, rest = text.partition("---\n")
    block, _, body = rest.partition("---\n")
    meta: dict[str, str] = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, body.strip()


@lru_cache(maxsize=None)
def load_persona(name: str) -> Persona:
    if name not in PERSONAS:
        raise UnknownPersona(f"persona desconocida: {name!r}. Válidas: {', '.join(PERSONAS)}")
    meta, body = _parse_front_matter((PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8"))
    raw_max = meta.get("max_insights", "null")
    return Persona(
        name=name,
        version=meta.get("version", "0"),
        max_insights=None if raw_max in ("null", "", "none") else int(raw_max),
        proposed_action=meta.get("proposed_action", "optional"),
        evidence=meta.get("evidence", "full"),
        assumptions=meta.get("assumptions", "visible"),
        confidence=meta.get("confidence", "label"),
        horizon=meta.get("horizon", ""),
        prompt=body,
    )


def persona_prompt(name: str) -> str:
    persona = load_persona(name)
    return (f"# Persona: {persona.name} (v{persona.version})\n\n{persona.prompt}\n\n"
            f"Horizonte sugerido: {persona.horizon}.")


FALLBACK_ACTION = ProposedAction(
    what=("Confirmar el hallazgo antes de actuar: reproducir la misma consulta sobre la ventana "
          "siguiente y comprobar que el patrón persiste."),
    effort="low",
    estimated_impact="No estimable con los datos disponibles.",
    validation="El patrón se mantiene en la ventana siguiente con el mismo signo y magnitud.",
)


def _summarize_evidence(insight: Insight) -> Insight:
    """Evidencia resumida: se deduplica por (métrica, dimensiones). Ningún valor cambia.

    Se reserva sitio para una evidencia de cada fuente citada: si el insight
    afirma una causa, su versión resumida tiene que seguir enseñando las dos
    fuentes que la sostienen, o la afirmación deja de leerse como lo que es.
    """
    unique: list = []
    seen: set[tuple] = set()
    for evidence in insight.evidence:
        key = (evidence.metric, tuple(sorted(evidence.dimensions.items())), evidence.window)
        if key in seen:
            continue
        seen.add(key)
        unique.append(evidence)

    first_by_source: list = []
    sources: set[str] = set()
    for evidence in unique:
        if evidence.source not in sources:
            sources.add(evidence.source)
            first_by_source.append(evidence)

    limit = max(SUMMARY_EVIDENCE_LIMIT, len(first_by_source))
    chosen = list(first_by_source)
    for evidence in unique:
        if len(chosen) >= limit:
            break
        if evidence not in chosen:
            chosen.append(evidence)
    kept = [e for e in unique if e in chosen]
    return insight.model_copy(update={"evidence": kept or insight.evidence[:1]})


def _sample_size_note(insight: Insight, observations_by_call: dict[str, dict]) -> str | None:
    sizes: list[str] = []
    for evidence in insight.evidence:
        key = (evidence.tool_call_id, evidence.metric)
        n = observations_by_call.get(key)
        if n:
            sizes.append(f"{evidence.metric} n={n}")
    if not sizes:
        return None
    return "Tamaños de muestra: " + "; ".join(dict.fromkeys(sizes)) + "."


def apply_persona(
    insights: list[Insight],
    persona: str | Persona,
    *,
    sample_sizes: dict[tuple[str, str], int] | None = None,
) -> list[Insight]:
    """Proyecta los insights canónicos a la vista de una persona.

    Nunca recalcula una cifra: recorta, oculta o completa, y nada más.
    """
    p = persona if isinstance(persona, Persona) else load_persona(persona)
    out: list[Insight] = []
    for insight in insights[: p.max_insights] if p.max_insights else insights:
        projected = insight
        if p.evidence == "summary":
            projected = _summarize_evidence(projected)
        if p.proposed_action == "required" and projected.proposed_action is None:
            projected = projected.model_copy(update={
                "proposed_action": FALLBACK_ACTION,
                "assumptions": [*projected.assumptions,
                                "Acción propuesta por defecto: el análisis no arroja una acción directa."],
            })
        if p.confidence == "label_and_sample" and sample_sizes:
            note = _sample_size_note(projected, sample_sizes)
            if note:
                projected = projected.model_copy(update={"assumptions": [*projected.assumptions, note]})
        if p.assumptions == "hidden":
            projected = projected.model_copy(update={"assumptions": []})
        out.append(projected)
    return out


__all__ = [
    "DEFAULT_PERSONA",
    "PERSONAS",
    "Persona",
    "UnknownPersona",
    "apply_persona",
    "load_persona",
    "persona_prompt",
]
