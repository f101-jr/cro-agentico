"""GA4: calidad del dato, reintentos y zona horaria."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from connectors.base import FetchRequest, SourceUnavailableError, TransientSourceError
from connectors.ga4 import GA4Connector, GA4FixtureTransport, GA4Report
from semantic.schema import DataQuality, Device, Window


class FlakyTransport:
    """Devuelve 429 las primeras `fail_times` veces."""

    property_timezone = "Europe/Madrid"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.attempts = 0

    async def run_report(self, **kwargs) -> GA4Report:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise TransientSourceError("ga4", 429, "rate limited")
        return GA4Report(rows=[{"dimension_value": None, "sessions": 10, "users": 8, "views": 10,
                                "carts": 3, "checkouts": 2, "purchases": 1}],
                         last_complete_date=date.today())

    async def ping(self) -> GA4Report:
        return await self.run_report()


class StaleTransport:
    """Consolida con 48 h de retraso, como GA4 real."""

    property_timezone = "Europe/Madrid"

    async def run_report(self, **kwargs) -> GA4Report:
        return GA4Report(rows=[{"dimension_value": None, "sessions": 100, "users": 80, "views": 100,
                                "carts": 30, "checkouts": 20, "purchases": 10}],
                         last_complete_date=date.today() - timedelta(days=2))

    async def ping(self) -> GA4Report:
        return await self.run_report()


class SampledTransport(StaleTransport):
    async def run_report(self, **kwargs) -> GA4Report:
        report = await super().run_report()
        return report.model_copy(update={"sampled": True, "sample_size": 12_345,
                                         "last_complete_date": date.today()})


async def test_window_including_last_48h_is_partial():
    connector = GA4Connector(StaleTransport(), retry_delay=0.0)
    obs = await connector.fetch(FetchRequest(metric="sessions", window=Window.LAST_7D))
    assert obs[0].quality is DataQuality.PARTIAL
    assert "no consolidada" in obs[0].note


async def test_sampled_response_is_flagged():
    connector = GA4Connector(SampledTransport(), retry_delay=0.0)
    obs = await connector.fetch(FetchRequest(metric="sessions", window=Window.LAST_7D))
    assert obs[0].quality is DataQuality.SAMPLED


async def test_429_retries_three_times_then_source_unavailable():
    transport = FlakyTransport(fail_times=99)
    connector = GA4Connector(transport, retry_delay=0.0)
    with pytest.raises(SourceUnavailableError) as exc:
        await connector.fetch(FetchRequest(metric="sessions", window=Window.LAST_7D))
    assert transport.attempts == 3
    assert exc.value.source == "ga4"


async def test_429_recovers_within_three_attempts():
    transport = FlakyTransport(fail_times=2)
    connector = GA4Connector(transport, retry_delay=0.0)
    obs = await connector.fetch(FetchRequest(metric="sessions", window=Window.LAST_7D))
    assert transport.attempts == 3
    assert obs[0].value == 10


async def test_windows_are_resolved_in_property_timezone(ga4_transport):
    connector = GA4Connector(ga4_transport, retry_delay=0.0)
    start, end = connector.window_dates(Window.LAST_14D)
    assert (end - start).days == 13
    assert connector.timezone == "Europe/Madrid"


async def test_funnel_returns_volumes_and_rates(ga4_transport):
    connector = GA4Connector(ga4_transport, retry_delay=0.0)
    obs = await connector.fetch(FetchRequest(metric="funnel_steps", window=Window.LAST_14D,
                                             device=Device.MOBILE))
    volumes = {o.dimensions["step"]: o.value for o in obs if o.metric == "funnel_step_volume"}
    rates = {o.dimensions["step"]: o.value for o in obs if o.metric == "funnel_step_rate"}
    assert volumes["view"] == 51_400 and volumes["purchase"] == 948
    assert round(rates["checkout_to_purchase"], 4) == round(948 / 2591, 4)


async def test_unknown_dimension_raises_value_error(ga4_transport):
    connector = GA4Connector(ga4_transport, retry_delay=0.0)
    with pytest.raises(ValueError):
        await connector.fetch(FetchRequest(metric="traffic_by_dim", window=Window.LAST_7D,
                                           dimension="inventado"))
