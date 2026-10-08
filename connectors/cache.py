"""Caché de respuestas de tool.

En v1 es en memoria. La interfaz es asíncrona a propósito para poder sustituirla
por Redis sin tocar a ningún llamante.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol, runtime_checkable

TTL_CLOSED_DAYS_S = 24 * 3600  # métricas de días cerrados
TTL_INCLUDES_TODAY_S = 15 * 60  # métricas que incluyen hoy


@runtime_checkable
class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_s: float) -> None: ...
    async def clear(self) -> None: ...


class InMemoryCache:
    def __init__(self, *, clock=time.monotonic) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._clock = clock
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._data[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    async def set(self, key: str, value: Any, ttl_s: float) -> None:
        self._data[key] = (self._clock() + ttl_s, value)

    async def clear(self) -> None:
        self._data.clear()
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._data)


class NullCache:
    """Desactiva la caché sin cambiar el código llamante (útil en evals)."""

    hits = misses = 0

    async def get(self, key: str) -> Any | None:
        return None

    async def set(self, key: str, value: Any, ttl_s: float) -> None:
        return None

    async def clear(self) -> None:
        return None


def cache_key(tool: str, arguments: dict) -> str:
    """Clave estable por (tool, argumentos): el orden de las claves no cuenta."""
    return f"{tool}:{json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)}"


def ttl_for_window(window_end, today) -> float:
    """24 h si la ventana está cerrada; 15 min si alcanza el día de hoy."""
    return TTL_INCLUDES_TODAY_S if window_end >= today else TTL_CLOSED_DAYS_S


__all__ = [
    "Cache",
    "InMemoryCache",
    "NullCache",
    "TTL_CLOSED_DAYS_S",
    "TTL_INCLUDES_TODAY_S",
    "cache_key",
    "ttl_for_window",
]
