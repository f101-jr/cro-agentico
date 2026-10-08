"""Contrato HTTP."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import MAX_QUESTION_CHARS, create_app
from config import Settings


@pytest.fixture
def client(container):
    return TestClient(create_app(container))


def test_every_endpoint_returns_x_run_id(client):
    run_id = client.get("/insights").headers["X-Run-Id"]
    assert client.get("/connectors/health").headers["X-Run-Id"]
    assert client.get(f"/trace/{run_id}").headers["X-Run-Id"] == run_id
    assert client.get("/insights?persona=nope").headers["X-Run-Id"]
    assert client.post("/ask", json={"question": "¿Cuántos pedidos hubo?"}).headers["X-Run-Id"]


def test_invalid_persona_is_400_not_500(client):
    response = client.get("/insights?persona=cfo")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_persona"


def test_invalid_window_is_400(client):
    response = client.get("/insights?window=el_mes_pasado")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_window"


def test_question_too_long_is_422(client):
    response = client.post("/ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "question_too_long"


def test_unknown_run_and_insight_are_404(client):
    assert client.get("/trace/run_inventado").json()["error"]["code"] == "run_not_found"
    assert client.get("/evidence/no_existe").json()["error"]["code"] == "insight_not_found"


def test_degraded_run_returns_200(client):
    response = client.post("/ask", json={"question": "¿Qué producto deja más margen?"})
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["insights"][0]["degraded"] is True


def test_insights_endpoint_serves_the_dashboard(client):
    body = client.get("/insights?persona=ecommerce_manager&window=last_14d").json()
    assert body["window"] == "last_14d"
    assert body["insights"]
    assert all(i["proposed_action"] for i in body["insights"])


def test_evidence_endpoint_returns_observations_behind_the_insight(client):
    body = client.post("/ask", json={"question": "La conversión en móvil ha caído. ¿Por qué?"}).json()
    insight = body["insights"][0]
    evidence = client.get(f"/evidence/{insight['id']}").json()
    assert evidence["run_id"] == body["run_id"]
    assert len(evidence["evidence"]) == len(insight["evidence"])
    assert evidence["observations"]
    metrics = {o["metric"] for o in evidence["observations"]}
    assert {e["metric"] for e in evidence["evidence"]} <= metrics


def test_trace_endpoint_returns_the_full_trace(client):
    run_id = client.post("/ask", json={"question": "¿Cuántos pedidos hubo?"}).json()["run_id"]
    trace = client.get(f"/trace/{run_id}").json()
    assert trace["run_id"] == run_id
    assert trace["prompt_version"]
    assert trace["tool_calls"]
    assert trace["iterations"] >= 1


def test_connectors_health_lists_every_source(client):
    sources = client.get("/connectors/health").json()["sources"]
    assert {s["source"] for s in sources} == {"ga4", "shopify", "support"}


def test_rate_limit_returns_429():
    app = create_app(settings=Settings(mode="fixtures", trace_db=":memory:", rate_limit_per_minute=2))
    client = TestClient(app)
    codes = [client.get("/connectors/health").status_code for _ in range(4)]
    assert codes[:2] == [200, 200]
    assert codes[-1] == 429
    assert client.get("/connectors/health").json()["error"]["code"] == "rate_limited"


def test_error_body_has_the_single_format(client):
    body = client.get("/insights?persona=cfo").json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "run_id"}


def test_run_timeout_is_504(client, container, monkeypatch):
    from agent.guardrails import RunTimeout

    async def times_out(*args, **kwargs):
        raise RunTimeout("demasiado")

    monkeypatch.setattr(container.orchestrator, "run", times_out)
    response = client.post("/ask", json={"question": "¿Cuántos pedidos hubo?"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "run_timeout"


def test_all_sources_down_is_424():
    """Si ninguna tool ha podido responder y las fuentes no contestan, es 424."""
    from connectors.base import TransientSourceError
    from connectors.ga4 import GA4Connector
    from connectors.shopify import ShopifyConnector
    from connectors.support import SupportConnector
    from container import build_container
    from semantic.tools import ToolContext

    class Dead:
        property_timezone = "Europe/Madrid"

        async def run_report(self, **kwargs):
            raise TransientSourceError("ga4", 503, "caída")

        async def iter_orders(self, **kwargs):
            raise TransientSourceError("shopify", 503, "caída")
            yield {}

        async def iter_tickets(self, **kwargs):
            raise TransientSourceError("support", 503, "caída")
            yield {}

        async def ping(self):
            raise TransientSourceError("*", 503, "caída")

    container = build_container(Settings(mode="fixtures", trace_db=":memory:",
                                         rate_limit_per_minute=0, retry_delay_s=0.0))
    container.context = ToolContext(
        ga4=GA4Connector(Dead(), retry_delay=0.0),
        shopify=ShopifyConnector(Dead(), retry_delay=0.0),
        support=SupportConnector(Dead(), retry_delay=0.0),
    )
    container.orchestrator.context = container.context
    client = TestClient(create_app(container))
    response = client.post("/ask", json={"question": "¿Cuántos pedidos hubo?"})
    assert response.status_code == 424
    body = response.json()["error"]
    assert body["code"] == "source_unavailable" and body["run_id"]
