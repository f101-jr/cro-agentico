"""Configuración por variable de entorno. Ninguna credencial vive en el repo."""

from __future__ import annotations

import os
from dataclasses import dataclass

from semantic.schema import Window


@dataclass(frozen=True)
class Settings:
    mode: str = "fixtures"  # fixtures | live
    llm_provider: str = "scripted"  # scripted | vertex
    vertex_project: str | None = None
    vertex_region: str = "europe-west1"
    vertex_model: str = "claude-sonnet-5@20250929"
    llm_max_tokens: int = 4096

    ga4_property_id: str | None = None
    ga4_timezone: str = "Europe/Madrid"

    shopify_shop: str | None = None
    shopify_token: str | None = None
    shopify_api_version: str = "2024-10"

    support_base_url: str | None = None
    support_token: str = ""

    trace_db: str = "./traces.sqlite3"
    rate_limit_per_minute: int = 30
    default_window: Window = Window.LAST_14D
    retry_delay_s: float = 0.5

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            mode=os.getenv("CRO_MODE", "fixtures").strip().lower(),
            llm_provider=os.getenv("CRO_LLM_PROVIDER", "scripted").strip().lower(),
            vertex_project=os.getenv("CRO_VERTEX_PROJECT") or None,
            vertex_region=os.getenv("CRO_VERTEX_REGION", "europe-west1"),
            vertex_model=os.getenv("CRO_VERTEX_MODEL", "claude-sonnet-5@20250929"),
            llm_max_tokens=_int("CRO_LLM_MAX_TOKENS", 4096),
            ga4_property_id=os.getenv("CRO_GA4_PROPERTY_ID") or None,
            ga4_timezone=os.getenv("CRO_GA4_TIMEZONE", "Europe/Madrid"),
            shopify_shop=os.getenv("CRO_SHOPIFY_SHOP") or None,
            shopify_token=os.getenv("CRO_SHOPIFY_TOKEN") or None,
            shopify_api_version=os.getenv("CRO_SHOPIFY_API_VERSION", "2024-10"),
            support_base_url=os.getenv("CRO_SUPPORT_BASE_URL") or None,
            support_token=os.getenv("CRO_SUPPORT_TOKEN", ""),
            trace_db=os.getenv("CRO_TRACE_DB", "./traces.sqlite3"),
            rate_limit_per_minute=_int("CRO_RATE_LIMIT_PER_MINUTE", 30),
        )

    @property
    def fixtures(self) -> bool:
        return self.mode == "fixtures"


__all__ = ["Settings"]
