"""Si un fixture rompe una invariante, pytest falla."""

from __future__ import annotations

from connectors.windows import day_offsets
from fixtures.dataset import load_dataset
from fixtures.validate import validate
from semantic.schema import Window


def test_dataset_invariants_hold():
    problems = validate()
    assert problems == [], "\n".join(problems)


def test_scenario_table_mobile_reproduced():
    d = load_dataset()
    last = d.ga4_totals(day_offsets(Window.LAST_14D), "mobile")
    prev = d.ga4_totals(day_offsets(Window.PREV_14D), "mobile")
    assert (prev["sessions"], last["sessions"]) == (48_200, 51_400)

    orders_last = d.shopify_totals(day_offsets(Window.LAST_14D), "mobile")
    orders_prev = d.shopify_totals(day_offsets(Window.PREV_14D), "mobile")
    assert (orders_prev["orders"], orders_last["orders"]) == (1_012, 948)
    assert round(orders_last["revenue_eur"] / orders_last["orders"], 2) == 78.00
    assert round(orders_prev["revenue_eur"] / orders_prev["orders"], 2) == 71.00


def test_funnel_steps_are_monotonic_every_window_and_device():
    d = load_dataset()
    for window in Window:
        for device in ("all", "mobile", "desktop", "tablet"):
            g = d.ga4_totals(day_offsets(window), device)
            steps = [g["views"], g["carts"], g["checkouts"], g["purchases"]]
            assert steps == sorted(steps, reverse=True), f"{window}/{device}: {steps}"


def test_orders_equal_sum_of_buckets():
    d = load_dataset()
    for window in Window:
        s = d.shopify_totals(day_offsets(window), "all")
        assert s["orders"] == sum(b["orders"] for b in s["buckets"].values())
