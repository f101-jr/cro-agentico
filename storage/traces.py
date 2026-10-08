"""Persistencia de trazas: SQLite en v1, con interfaz sustituible.

Solo se persisten trazas y los insights que produjeron. Ningún dato de negocio
vive aquí más allá de las observaciones que respaldan una respuesta concreta.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from semantic.schema import Insight, Observation, Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    persona    TEXT NOT NULL,
    question   TEXT NOT NULL,
    started_at TEXT NOT NULL,
    trace      TEXT NOT NULL,
    insights   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS insights (
    insight_id TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);
"""


@runtime_checkable
class TraceStore(Protocol):
    async def save_run(self, trace: Trace, insights: list[Insight]) -> None: ...
    async def get_trace(self, run_id: str) -> Trace | None: ...
    async def get_insights(self, run_id: str) -> list[Insight]: ...
    async def find_insight(self, insight_id: str) -> tuple[Insight, Trace] | None: ...
    async def list_runs(self, limit: int = 20) -> list[Trace]: ...


class InMemoryTraceStore:
    def __init__(self) -> None:
        self._runs: dict[str, tuple[Trace, list[Insight]]] = {}

    async def save_run(self, trace: Trace, insights: list[Insight]) -> None:
        self._runs[trace.run_id] = (trace, list(insights))

    async def get_trace(self, run_id: str) -> Trace | None:
        entry = self._runs.get(run_id)
        return entry[0] if entry else None

    async def get_insights(self, run_id: str) -> list[Insight]:
        entry = self._runs.get(run_id)
        return list(entry[1]) if entry else []

    async def find_insight(self, insight_id: str) -> tuple[Insight, Trace] | None:
        for trace, insights in self._runs.values():
            for insight in insights:
                if insight.id == insight_id:
                    return insight, trace
        return None

    async def list_runs(self, limit: int = 20) -> list[Trace]:
        traces = [trace for trace, _ in self._runs.values()]
        return sorted(traces, key=lambda t: t.started_at, reverse=True)[:limit]


class SQLiteTraceStore:
    """SQLite síncrono ejecutado en hilo: la interfaz es async para poder
    sustituirlo por un almacén remoto sin tocar al orquestador."""

    def __init__(self, path: str | Path = "traces.sqlite3") -> None:
        self.path = str(path)
        self._lock = asyncio.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    async def save_run(self, trace: Trace, insights: list[Insight]) -> None:
        payload = (
            trace.run_id, trace.persona, trace.question, trace.started_at.isoformat(),
            trace.model_dump_json(), json.dumps([i.model_dump(mode="json") for i in insights],
                                                ensure_ascii=False),
        )
        ids = [(i.id, trace.run_id) for i in insights]

        def _write() -> None:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO runs (run_id, persona, question, started_at, trace, insights)"
                    " VALUES (?, ?, ?, ?, ?, ?)", payload,
                )
                conn.executemany("INSERT OR REPLACE INTO insights (insight_id, run_id) VALUES (?, ?)", ids)

        async with self._lock:
            await asyncio.to_thread(_write)

    async def get_trace(self, run_id: str) -> Trace | None:
        def _read():
            with self._connect() as conn:
                return conn.execute("SELECT trace FROM runs WHERE run_id = ?", (run_id,)).fetchone()

        row = await asyncio.to_thread(_read)
        return Trace.model_validate_json(row["trace"]) if row else None

    async def get_insights(self, run_id: str) -> list[Insight]:
        def _read():
            with self._connect() as conn:
                return conn.execute("SELECT insights FROM runs WHERE run_id = ?", (run_id,)).fetchone()

        row = await asyncio.to_thread(_read)
        return [Insight.model_validate(i) for i in json.loads(row["insights"])] if row else []

    async def find_insight(self, insight_id: str) -> tuple[Insight, Trace] | None:
        def _read():
            with self._connect() as conn:
                return conn.execute(
                    "SELECT r.trace AS trace, r.insights AS insights FROM insights i "
                    "JOIN runs r ON r.run_id = i.run_id WHERE i.insight_id = ?", (insight_id,),
                ).fetchone()

        row = await asyncio.to_thread(_read)
        if not row:
            return None
        trace = Trace.model_validate_json(row["trace"])
        for raw in json.loads(row["insights"]):
            if raw.get("id") == insight_id:
                return Insight.model_validate(raw), trace
        return None

    async def list_runs(self, limit: int = 20) -> list[Trace]:
        def _read():
            with self._connect() as conn:
                return conn.execute(
                    "SELECT trace FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()

        rows = await asyncio.to_thread(_read)
        return [Trace.model_validate_json(r["trace"]) for r in rows]


def observations_for(trace: Trace, insight: Insight) -> list[Observation]:
    """Observaciones que respaldan un insight, en el orden en que se citaron."""
    by_id = {call.id: call for call in trace.tool_calls}
    out: list[Observation] = []
    for evidence in insight.evidence:
        call = by_id.get(evidence.tool_call_id)
        if call is None:
            continue
        out.extend(obs for obs in call.observations if obs.metric == evidence.metric)
    seen: set[str] = set()
    unique: list[Observation] = []
    for obs in out:
        key = obs.model_dump_json(exclude={"fetched_at"})
        if key not in seen:
            seen.add(key)
            unique.append(obs)
    return unique


__all__ = ["InMemoryTraceStore", "SQLiteTraceStore", "TraceStore", "observations_for"]
