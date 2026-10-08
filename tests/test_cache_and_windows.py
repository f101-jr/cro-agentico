"""Caché con TTL por ventana y resolución de ventanas."""

from __future__ import annotations

from datetime import date, timedelta

from connectors.cache import (
    TTL_CLOSED_DAYS_S,
    TTL_INCLUDES_TODAY_S,
    InMemoryCache,
    cache_key,
    ttl_for_window,
)
from connectors.windows import COMPARISON_PAIR, day_offsets, resolve_window, window_days
from semantic.schema import Window


def test_ttl_is_15_minutes_when_the_window_reaches_today():
    today = date(2026, 5, 10)
    assert ttl_for_window(today, today) == TTL_INCLUDES_TODAY_S
    assert ttl_for_window(today - timedelta(days=1), today) == TTL_CLOSED_DAYS_S


def test_cache_key_ignores_argument_order():
    assert cache_key("t", {"a": 1, "b": 2}) == cache_key("t", {"b": 2, "a": 1})


async def test_cache_expires():
    clock = [0.0]
    cache = InMemoryCache(clock=lambda: clock[0])
    await cache.set("k", {"v": 1}, ttl_s=10)
    assert await cache.get("k") == {"v": 1}
    clock[0] = 11
    assert await cache.get("k") is None


def test_windows_never_include_today():
    today = date(2026, 5, 10)
    for window in Window:
        start, end = resolve_window(window, today=today)
        assert end < today
        assert start <= end


def test_window_lengths_and_pairs():
    assert window_days(Window.LAST_7D) == 7
    assert window_days(Window.LAST_30D) == 30
    assert COMPARISON_PAIR[Window.LAST_14D] is Window.PREV_14D
    assert day_offsets(Window.PREV_14D) == list(range(15, 29))


def test_previous_windows_do_not_overlap_current_ones():
    today = date(2026, 5, 10)
    for current, previous in COMPARISON_PAIR.items():
        c_start, c_end = resolve_window(current, today=today)
        p_start, p_end = resolve_window(previous, today=today)
        assert p_end < c_start


def test_same_period_last_year_is_a_year_back():
    today = date(2026, 5, 10)
    start, end = resolve_window(Window.SAME_PERIOD_LAST_YEAR, today=today)
    current_start, current_end = resolve_window(Window.LAST_14D, today=today)
    assert (current_end - end).days == 364
    assert (end - start).days == 13
