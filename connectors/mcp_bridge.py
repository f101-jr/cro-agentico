"""Puente MCP.

Envuelve servidores MCP y los expone como `Connector`. **El LLM nunca recibe las
tools del servidor MCP**: el puente las traduce a `Observation` y la capa
semántica decide qué se expone.

Un servidor sin `MCPMapping` declarado no se registra. Fallar en arranque es
preferible a exponer al modelo una superficie desconocida.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from connectors.base import Capability, FetchRequest, HealthStatus, strip_pii, with_retries
from connectors.windows import resolve_window
from semantic.schema import DataQuality, Device, Observation, Unit, Window


class MCPMappingError(RuntimeError):
    """Se lanza en arranque. Nunca se captura para “seguir tirando”."""


class MCPToolMapping(BaseModel):
    """Traducción de UNA tool del servidor MCP a UNA métrica canónica."""

    tool: str
    metric: str
    unit: Unit = "count"
    value_path: str = "value"  # ruta con puntos; los enteros indexan listas
    argument_map: dict[str, str] = Field(default_factory=dict)  # canónico -> nombre en el servidor
    static_arguments: dict[str, Any] = Field(default_factory=dict)
    dimension_paths: dict[str, str] = Field(default_factory=dict)
    sample_size_path: str | None = None
    note: str | None = None
    windows: list[Window] = Field(default_factory=lambda: list(Window))


class MCPMapping(BaseModel):
    server: str
    source: str  # nombre canónico de la fuente en las Observation
    tools: list[MCPToolMapping] = Field(min_length=1)

    def by_metric(self) -> dict[str, MCPToolMapping]:
        return {t.metric: t for t in self.tools}


@runtime_checkable
class MCPClient(Protocol):
    async def list_tools(self) -> list[str]: ...
    async def call_tool(self, name: str, arguments: dict) -> dict: ...
    async def ping(self) -> bool: ...


class MCPConnector:
    """Conector de solo lectura respaldado por un servidor MCP."""

    def __init__(self, server_uri: str, mapping: MCPMapping, *, client: MCPClient,
                 timezone_name: str = "Europe/Madrid", retry_delay: float = 0.5) -> None:
        if mapping is None:  # defensa explícita: el tipo no basta si llega None en runtime
            raise MCPMappingError(f"servidor MCP {server_uri!r} sin MCPMapping declarado")
        self.server_uri = server_uri
        self.mapping = mapping
        self.client = client
        self.name = mapping.source
        self.timezone_name = timezone_name
        self._retry_delay = retry_delay

    async def verify(self) -> None:
        """Arranque: toda tool del servidor debe estar mapeada. Si no, se aborta."""
        exposed = set(await self.client.list_tools())
        mapped = {t.tool for t in self.mapping.tools}
        missing = mapped - exposed
        unmapped = exposed - mapped
        if missing:
            raise MCPMappingError(
                f"servidor MCP {self.server_uri!r}: el mapping declara tools que el servidor no "
                f"expone: {sorted(missing)}"
            )
        if unmapped:
            raise MCPMappingError(
                f"servidor MCP {self.server_uri!r}: tools sin mapping declarado: {sorted(unmapped)}. "
                "Declara su métrica canónica en MCPMapping o retira la tool del servidor: no se "
                "expondrá una superficie desconocida al modelo."
            )

    def capabilities(self) -> list[Capability]:
        return [
            Capability(metric=t.metric, dimensions=sorted(t.dimension_paths), windows=t.windows)
            for t in self.mapping.tools
        ]

    async def fetch(self, request: FetchRequest) -> list[Observation]:
        tool = self.mapping.by_metric().get(request.metric)
        if tool is None:
            raise ValueError(f"métrica no mapeada en el servidor MCP {self.server_uri!r}: {request.metric}")

        from zoneinfo import ZoneInfo

        start, end = resolve_window(request.window, tz=ZoneInfo(self.timezone_name))
        canonical = {
            "window": request.window.value,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "device": request.device.value,
            "dimension": request.dimension or "",
            "limit": request.limit or 0,
        }
        arguments = dict(tool.static_arguments)
        for canonical_name, remote_name in tool.argument_map.items():
            if canonical_name in canonical:
                arguments[remote_name] = canonical[canonical_name]

        payload = await with_retries(
            lambda: self.client.call_tool(tool.tool, arguments),
            source=self.name, base_delay=self._retry_delay,
        )
        safe = strip_pii(payload if isinstance(payload, dict) else {"value": payload})
        value = _extract(safe, tool.value_path)
        if value is None:
            raise ValueError(f"la tool MCP {tool.tool!r} no devolvió {tool.value_path!r}")

        dimensions = {"device": request.device.value}
        for name, path in tool.dimension_paths.items():
            found = _extract(safe, path)
            if found is not None:
                dimensions[name] = str(found)
        sample = _extract(safe, tool.sample_size_path) if tool.sample_size_path else None

        return [Observation(
            metric=tool.metric, value=float(value), unit=tool.unit, window=request.window,
            window_start=start, window_end=end, dimensions=dimensions, source=self.name,
            quality=DataQuality.EXACT, sample_size=int(sample) if sample is not None else None,
            note=tool.note, fetched_at=datetime.now(timezone.utc),
        )]

    async def health(self) -> HealthStatus:
        started = time.perf_counter()
        try:
            ok = await with_retries(self.client.ping, source=self.name, base_delay=self._retry_delay)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(source=self.name, reachable=False,
                                latency_ms=int((time.perf_counter() - started) * 1000), message=str(exc))
        return HealthStatus(source=self.name, reachable=bool(ok), last_complete_date=_today(self.timezone_name),
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            message=f"servidor MCP {self.server_uri} · {len(self.mapping.tools)} tools mapeadas")


async def build_mcp_connectors(
    servers: dict[str, MCPMapping | None],
    clients: dict[str, MCPClient],
    *,
    verify: bool = True,
) -> list[MCPConnector]:
    """Registro de servidores MCP. Falla en arranque ante cualquier hueco."""
    connectors: list[MCPConnector] = []
    for uri, mapping in servers.items():
        if mapping is None:
            raise MCPMappingError(
                f"servidor MCP {uri!r} sin MCPMapping declarado: no se registra. Declara a qué "
                "métrica canónica corresponde cada tool antes de arrancar."
            )
        client = clients.get(uri)
        if client is None:
            raise MCPMappingError(f"servidor MCP {uri!r} sin cliente configurado")
        connector = MCPConnector(uri, mapping, client=client)
        if verify:
            await connector.verify()
        connectors.append(connector)
    return connectors


def _extract(payload: Any, path: str | None) -> Any:
    if not path:
        return None
    node = payload
    for part in path.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            return None
    return node


def _today(tz_name: str) -> date:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(tz_name)).date()


__all__ = [
    "MCPClient",
    "MCPConnector",
    "MCPMapping",
    "MCPMappingError",
    "MCPToolMapping",
    "build_mcp_connectors",
]
