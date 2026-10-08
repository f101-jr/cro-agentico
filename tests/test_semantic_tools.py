"""Contrato de las tools expuestas al modelo."""

from __future__ import annotations

from semantic.tools import MAX_ROWS, REGISTRY, ToolResult, execute_tool, tool_schemas
from semantic.truth import DERIVED_METRICS


def test_there_are_exactly_eight_tools():
    assert len(REGISTRY) == 8
    assert set(REGISTRY) == {
        "get_funnel_metrics", "get_traffic_breakdown", "get_orders_summary",
        "get_product_performance", "get_conversion_rate", "compare_windows",
        "get_support_signal", "get_data_health",
    }


def test_schemas_are_serializable_and_described():
    for schema in tool_schemas():
        assert schema["description"].strip()
        assert schema["input_schema"]["type"] == "object"


async def test_conversion_rate_always_carries_composition_note(ctx):
    result = await execute_tool(ctx, "get_conversion_rate", {"window": "last_14d", "device": "mobile"})
    derived = [o for o in result.observations if o.metric == "conversion_rate"]
    assert derived, "la tool derivada no ha devuelto la métrica"
    note = derived[0].note or ""
    assert DERIVED_METRICS["conversion_rate"]["note"] in note
    assert DERIVED_METRICS["conversion_rate"]["formula"] in note
    assert derived[0].source == "derived"


async def test_no_tool_returns_more_than_25_rows(ctx):
    calls = [
        ("get_traffic_breakdown", {"window": "last_14d", "dimension": "source_medium", "limit": 25}),
        ("get_orders_summary", {"window": "last_14d", "dimension": "channel"}),
        ("get_product_performance", {"window": "last_14d", "metric": "units", "limit": 25}),
        ("get_support_signal", {"window": "last_14d"}),
    ]
    for name, args in calls:
        result = await execute_tool(ctx, name, args)
        rows = {tuple(sorted(o.dimensions.items())) for o in result.observations}
        assert len(rows) <= MAX_ROWS, name


async def test_truncation_is_declared_in_the_note(ctx):
    result = await execute_tool(ctx, "get_traffic_breakdown",
                                {"window": "last_14d", "dimension": "source_medium", "limit": 2})
    rows = {tuple(sorted(o.dimensions.items())) for o in result.observations}
    assert len(rows) == 2
    assert result.truncated
    assert all("truncado" in (o.note or "") for o in result.observations)


async def test_limit_above_hard_max_is_rejected_with_guidance(ctx):
    result = await execute_tool(ctx, "get_traffic_breakdown",
                                {"window": "last_14d", "dimension": "channel", "limit": 500})
    assert result.ok is False
    assert "limit" in result.guidance


async def test_enum_argument_error_is_text_not_exception(ctx):
    result = await execute_tool(ctx, "get_orders_summary", {"window": "last_14d", "dimension": "luna"})
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "Argumentos inválidos" in result.guidance
    assert "aov_bucket" in result.guidance


async def test_unknown_tool_is_text_not_exception(ctx):
    result = await execute_tool(ctx, "borrar_pedidos", {})
    assert result.ok is False and "No existe la tool" in result.guidance


async def test_thin_sample_returns_guidance_not_numbers(ctx):
    result = await execute_tool(ctx, "get_conversion_rate", {"window": "last_7d", "device": "tablet"})
    assert result.ok is False
    assert result.observations == []
    assert "Sin datos suficientes" in result.guidance and "n=12" in result.guidance


async def test_identical_call_within_ttl_does_not_touch_the_source(ctx, shopify_transport):
    args = {"window": "last_14d", "dimension": "none"}
    await execute_tool(ctx, "get_orders_summary", args)
    calls_after_first = shopify_transport.calls
    await execute_tool(ctx, "get_orders_summary", dict(reversed(list(args.items()))))
    assert shopify_transport.calls == calls_after_first


async def test_compare_windows_does_all_arithmetic_in_python(ctx):
    result = await execute_tool(ctx, "compare_windows", {
        "metric": "conversion_rate", "window_a": "last_14d", "window_b": "prev_14d", "device": "mobile",
    })
    values = {o.metric: o.value for o in result.observations}
    a = next(o.value for o in result.observations
             if o.metric == "conversion_rate" and o.window == "last_14d")
    b = next(o.value for o in result.observations
             if o.metric == "conversion_rate" and o.window == "prev_14d")
    assert values["conversion_rate_outside_typical_variation"] == 1.0
    assert round(values["conversion_rate_delta_abs"], 6) == round(a - b, 6)
    assert round(values["conversion_rate_delta_rel"], 6) == round((a - b) / b, 6)


async def test_compare_windows_rejects_unsupported_metric(ctx):
    result = await execute_tool(ctx, "compare_windows", {
        "metric": "margin", "window_a": "last_14d", "window_b": "prev_14d",
    })
    assert result.ok is False and "COGS" in result.guidance


async def test_data_health_reports_every_source(ctx):
    result = await execute_tool(ctx, "get_data_health", {}, use_cache=False)
    assert {h.source for h in result.health} == {"ga4", "shopify", "support"}
    assert all(h.reachable for h in result.health)
