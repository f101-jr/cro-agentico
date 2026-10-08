"""Un servidor MCP sin mapping no arranca."""

from __future__ import annotations

import pytest

from connectors.base import FetchRequest
from connectors.mcp_bridge import (
    MCPConnector,
    MCPMapping,
    MCPMappingError,
    MCPToolMapping,
    build_mcp_connectors,
)
from semantic.schema import Window


class StubClient:
    def __init__(self, tools: list[str], payload: dict | None = None) -> None:
        self._tools = tools
        self._payload = payload or {"result": {"total": 42, "n": 100, "segment": "premium"}}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[str]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self._payload

    async def ping(self) -> bool:
        return True


MAPPING = MCPMapping(
    server="reviews", source="reviews",
    tools=[MCPToolMapping(tool="get_reviews", metric="review_volume", unit="count",
                          value_path="result.total", sample_size_path="result.n",
                          dimension_paths={"segment": "result.segment"},
                          argument_map={"start_date": "from", "end_date": "to"})],
)


async def test_server_without_mapping_fails_at_startup():
    with pytest.raises(MCPMappingError) as exc:
        await build_mcp_connectors({"reviews": None}, {"reviews": StubClient(["get_reviews"])})
    assert "sin MCPMapping declarado" in str(exc.value)


async def test_server_with_unmapped_tool_fails_at_startup():
    client = StubClient(["get_reviews", "delete_review"])
    with pytest.raises(MCPMappingError) as exc:
        await build_mcp_connectors({"reviews": MAPPING}, {"reviews": client})
    assert "delete_review" in str(exc.value)


async def test_mapped_server_registers_and_maps_to_observations():
    client = StubClient(["get_reviews"])
    connectors = await build_mcp_connectors({"reviews": MAPPING}, {"reviews": client})
    assert len(connectors) == 1
    obs = await connectors[0].fetch(FetchRequest(metric="review_volume", window=Window.LAST_7D))
    assert obs[0].value == 42 and obs[0].source == "reviews"
    assert obs[0].sample_size == 100 and obs[0].dimensions["segment"] == "premium"
    # el LLM nunca ve el nombre de la tool del servidor: solo la métrica canónica
    assert client.calls[0][0] == "get_reviews"
    assert set(client.calls[0][1]) == {"from", "to"}


async def test_unmapped_metric_is_rejected():
    connector = MCPConnector("reviews", MAPPING, client=StubClient(["get_reviews"]))
    with pytest.raises(ValueError):
        await connector.fetch(FetchRequest(metric="lo_que_sea", window=Window.LAST_7D))
