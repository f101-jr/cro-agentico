"""Ensamblado del sistema: qué transporte, qué proveedor, qué almacén.

`CRO_MODE=fixtures` sustituye **solo** la capa de transporte de los conectores.
La capa semántica y el orquestador son exactamente los mismos en ambos modos:
si una demo pasa en fixtures y falla en real, el problema está en la fuente, no
en el razonamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.orchestrator import Orchestrator
from agent.providers.base import LLMProvider
from config import Settings
from connectors.cache import Cache, InMemoryCache
from connectors.ga4 import GA4ApiTransport, GA4Connector, GA4FixtureTransport
from connectors.mcp_bridge import MCPClient, MCPConnector, MCPMapping, build_mcp_connectors
from connectors.shopify import ShopifyConnector, ShopifyFixtureTransport, ShopifyGraphQLTransport
from connectors.support import SupportConnector, SupportFixtureTransport, SupportHttpTransport
from semantic.tools import ToolContext
from storage.traces import InMemoryTraceStore, SQLiteTraceStore, TraceStore


@dataclass
class Container:
    settings: Settings
    context: ToolContext
    provider: LLMProvider
    store: TraceStore
    orchestrator: Orchestrator


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "vertex":
        from agent.providers.vertex import VertexClaudeProvider

        return VertexClaudeProvider(model=settings.vertex_model, project_id=settings.vertex_project,
                                    region=settings.vertex_region)
    from agent.providers.scripted import ScriptedProvider

    return ScriptedProvider(default_window=settings.default_window)


def build_context(settings: Settings, *, cache: Cache | None = None,
                  extra_connectors: list | None = None) -> ToolContext:
    if settings.fixtures:
        ga4 = GA4Connector(GA4FixtureTransport(property_timezone=settings.ga4_timezone),
                           retry_delay=settings.retry_delay_s)
        shopify = ShopifyConnector(ShopifyFixtureTransport(timezone_name=settings.ga4_timezone),
                                   retry_delay=settings.retry_delay_s, timezone_name=settings.ga4_timezone)
        support = SupportConnector(SupportFixtureTransport(timezone_name=settings.ga4_timezone),
                                   retry_delay=settings.retry_delay_s, timezone_name=settings.ga4_timezone)
    else:
        ga4 = GA4Connector(GA4ApiTransport(settings.ga4_property_id, property_timezone=settings.ga4_timezone),
                           retry_delay=settings.retry_delay_s)
        shopify = ShopifyConnector(
            ShopifyGraphQLTransport(settings.shopify_shop, settings.shopify_token,
                                    api_version=settings.shopify_api_version),
            retry_delay=settings.retry_delay_s, timezone_name=settings.ga4_timezone,
        )
        transport = (SupportHttpTransport(settings.support_base_url, settings.support_token)
                     if settings.support_base_url else SupportFixtureTransport(timezone_name=settings.ga4_timezone))
        support = SupportConnector(transport, retry_delay=settings.retry_delay_s,
                                   timezone_name=settings.ga4_timezone)

    return ToolContext(ga4=ga4, shopify=shopify, support=support,
                       cache=cache or InMemoryCache(), extra=list(extra_connectors or []),
                       timezone_name=settings.ga4_timezone)


def build_store(settings: Settings) -> TraceStore:
    return InMemoryTraceStore() if settings.trace_db in (":memory:", "") else SQLiteTraceStore(settings.trace_db)


def build_container(settings: Settings | None = None, *, cache: Cache | None = None,
                    extra_connectors: list | None = None) -> Container:
    settings = settings or Settings.from_env()
    context = build_context(settings, cache=cache, extra_connectors=extra_connectors)
    provider = build_provider(settings)
    store = build_store(settings)
    orchestrator = Orchestrator(provider, context, store=store, max_tokens=settings.llm_max_tokens)
    return Container(settings=settings, context=context, provider=provider, store=store,
                     orchestrator=orchestrator)


async def register_mcp_servers(
    context: ToolContext,
    servers: dict[str, MCPMapping | None],
    clients: dict[str, MCPClient],
) -> list[MCPConnector]:
    """Registra servidores MCP como conectores adicionales. Falla en arranque
    si alguno no declara mapping."""
    connectors = await build_mcp_connectors(servers, clients)
    context.extra.extend(connectors)
    return connectors


__all__ = ["Container", "build_container", "build_context", "build_provider", "build_store",
           "register_mcp_servers"]
