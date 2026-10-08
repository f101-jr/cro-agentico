"""Piezas comunes de los tests. Todo corre contra fixtures y sin red."""

from __future__ import annotations

import pytest

from config import Settings
from connectors.cache import InMemoryCache
from connectors.ga4 import GA4Connector, GA4FixtureTransport
from connectors.shopify import ShopifyConnector, ShopifyFixtureTransport
from connectors.support import SupportConnector, SupportFixtureTransport
from container import build_container
from semantic.tools import ToolContext


@pytest.fixture
def ga4_transport() -> GA4FixtureTransport:
    return GA4FixtureTransport()


@pytest.fixture
def shopify_transport() -> ShopifyFixtureTransport:
    return ShopifyFixtureTransport()


@pytest.fixture
def support_transport() -> SupportFixtureTransport:
    return SupportFixtureTransport()


@pytest.fixture
def connectors(ga4_transport, shopify_transport, support_transport):
    return (
        GA4Connector(ga4_transport, retry_delay=0.0),
        ShopifyConnector(shopify_transport, retry_delay=0.0),
        SupportConnector(support_transport, retry_delay=0.0),
    )


@pytest.fixture
def ctx(connectors) -> ToolContext:
    ga4, shopify, support = connectors
    return ToolContext(ga4=ga4, shopify=shopify, support=support, cache=InMemoryCache())


@pytest.fixture
def settings() -> Settings:
    return Settings(mode="fixtures", llm_provider="scripted", trace_db=":memory:",
                    rate_limit_per_minute=0)


@pytest.fixture
def container(settings):
    return build_container(settings)
