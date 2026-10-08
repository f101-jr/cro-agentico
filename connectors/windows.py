"""Resolución de `Window` a fechas concretas.

Las ventanas se resuelven **en el conector**, en la zona horaria de la fuente,
nunca en UTC y nunca en el LLM. Este módulo es la implementación compartida por
los conectores; nadie más debe inventar su propia aritmética de fechas.

Convenio, documentado también en el README:
  - El día de hoy NUNCA entra en una ventana: el último día de `last_*` es ayer.
  - `last_7d`  = [hoy-7,  hoy-1]      `prev_7d`  = [hoy-14, hoy-8]
  - `last_14d` = [hoy-14, hoy-1]      `prev_14d` = [hoy-28, hoy-15]
  - `last_30d` = [hoy-30, hoy-1]      `prev_30d` = [hoy-60, hoy-31]
  - `same_period_last_year` = `last_14d` desplazada 364 días (mismo día de la
    semana), es decir [hoy-378, hoy-365].
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo

from semantic.schema import Window

DEFAULT_TZ = ZoneInfo(os.getenv("CRO_DEFAULT_TIMEZONE", "Europe/Madrid"))

# offset del inicio y del fin respecto a hoy, en días (ambos inclusive)
_OFFSETS: dict[Window, tuple[int, int]] = {
    Window.LAST_7D: (7, 1),
    Window.PREV_7D: (14, 8),
    Window.LAST_14D: (14, 1),
    Window.PREV_14D: (28, 15),
    Window.LAST_30D: (30, 1),
    Window.PREV_30D: (60, 31),
    Window.SAME_PERIOD_LAST_YEAR: (14 + 364, 1 + 364),
}

# Pareja canónica de comparación: para cada ventana "actual", su periodo previo.
COMPARISON_PAIR: dict[Window, Window] = {
    Window.LAST_7D: Window.PREV_7D,
    Window.LAST_14D: Window.PREV_14D,
    Window.LAST_30D: Window.PREV_30D,
}


def today_in(tz: tzinfo | None = None) -> date:
    """Hoy en la zona horaria de la fuente (no en UTC)."""
    return datetime.now(tz or DEFAULT_TZ).date()


def resolve_window(window: Window, *, tz: tzinfo | None = None, today: date | None = None) -> tuple[date, date]:
    """Devuelve (start, end) inclusive para la ventana, en la zona horaria dada."""
    ref = today or today_in(tz)
    start_off, end_off = _OFFSETS[Window(window)]
    return ref - timedelta(days=start_off), ref - timedelta(days=end_off)


def window_days(window: Window) -> int:
    start_off, end_off = _OFFSETS[Window(window)]
    return start_off - end_off + 1


def day_offsets(window: Window) -> list[int]:
    """Offsets de día (1 = ayer) que componen la ventana. Base de los fixtures."""
    start_off, end_off = _OFFSETS[Window(window)]
    return list(range(end_off, start_off + 1))


def offset_to_date(offset: int, *, tz: tzinfo | None = None, today: date | None = None) -> date:
    ref = today or today_in(tz)
    return ref - timedelta(days=offset)


def date_to_offset(d: date, *, tz: tzinfo | None = None, today: date | None = None) -> int:
    ref = today or today_in(tz)
    return (ref - d).days


__all__ = [
    "COMPARISON_PAIR",
    "DEFAULT_TZ",
    "date_to_offset",
    "day_offsets",
    "offset_to_date",
    "resolve_window",
    "today_in",
    "window_days",
]
