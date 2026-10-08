"""Soporte: la clasificación por tema es código, no modelo."""

from __future__ import annotations

import pytest

from connectors.base import FetchRequest
from connectors.support import classify_topic
from semantic.schema import Window


@pytest.mark.parametrize("subject,expected", [
    ("¿Por qué me cobran envío si antes era gratis?", "shipping"),
    ("Me han cobrado portes sin avisar", "shipping"),
    ("Quiero tramitar una devolución", "returns"),
    ("¿Cuánto tarda el reembolso?", "returns"),
    ("El pago con tarjeta me da error", "payment"),
    ("No puedo acceder a mi cuenta", "account"),
    ("¿Qué talla es la sudadera con capucha?", "product"),
    ("Hola", "other"),
])
def test_keyword_classification(subject, expected):
    assert classify_topic(subject) == expected


def test_classification_ignores_accents_and_case():
    assert classify_topic("GASTOS DE ENVIO muy altos") == "shipping"


async def test_shipping_tickets_tripled_in_the_scenario(connectors):
    _, _, support = connectors
    last = await support.fetch(FetchRequest(metric="support_tickets", window=Window.LAST_14D))
    prev = await support.fetch(FetchRequest(metric="support_tickets", window=Window.PREV_14D))
    n_last = next(o.value for o in last if o.dimensions["topic"] == "shipping")
    n_prev = next(o.value for o in prev if o.dimensions["topic"] == "shipping")
    assert round(n_last / n_prev, 1) == 3.2


async def test_topic_filter_returns_only_that_topic(connectors):
    _, _, support = connectors
    obs = await support.fetch(FetchRequest(metric="support_tickets", window=Window.LAST_14D,
                                           filters={"topic": "payment"}))
    assert {o.dimensions["topic"] for o in obs} == {"payment"}
