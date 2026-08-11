"""Broker-bound price quantization (hotfix 2026-08-11, Alpaca 42210000).

The single seam every exit-policy / thesis order price passes through:
>= $1 → penny grid, < $1 → 4dp grid; "up" for breakeven-floor / TP
components (invariant-preserving), "down" for trail-derived protective
stops (semantically identical, always broker-valid).
"""

from __future__ import annotations

import pytest

from execution.quantize import quantize_price


def test_subpenny_entry_ceils_to_next_cent() -> None:
    # The ATRC price: avg fill 37.3297 → floor component 37.33.
    assert quantize_price(37.3297, direction="up") == 37.33


def test_subpenny_trail_floors_to_cent() -> None:
    assert quantize_price(37.3297, direction="down") == 37.32
    assert quantize_price(105.4975, direction="down") == 105.49


def test_clean_two_dp_price_is_identity_both_directions() -> None:
    # PKE escaped the incident because its fill was exactly 2dp — a
    # clean price must never shift a cent in either direction, even
    # though 37.33 has no exact binary representation.
    for price in (37.33, 24.85, 100.0, 1.00):
        assert quantize_price(price, direction="up") == price
        assert quantize_price(price, direction="down") == price


def test_float_noise_from_reconstructed_price_does_not_drop_a_cent() -> None:
    # entry − risk round-trips a clean 2dp stop through floats with
    # drift (6.7791 − (6.7791 − 2.28) = 2.2799999999999994); the 1e-6
    # pre-round keeps the directional floor from eating a cent.
    reconstructed = 6.7791 - (6.7791 - 2.28)
    assert reconstructed != 2.28  # the drift is real
    assert quantize_price(reconstructed, direction="down") == 2.28
    # And the mirror case: upward drift must not cost the ceil a cent.
    reconstructed_up = 152.9911 - (152.9911 - 39.77)
    assert reconstructed_up != 39.77
    assert quantize_price(reconstructed_up, direction="up") == 39.77


def test_sub_dollar_prices_use_four_dp_grid() -> None:
    # Alpaca allows $0.0001 increments under $1 — defensive only (the
    # universe price floor is $5) but handled, not assumed away.
    assert quantize_price(0.45237, direction="down") == 0.4523
    assert quantize_price(0.45237, direction="up") == 0.4524
    assert quantize_price(0.4523, direction="down") == 0.4523
    assert quantize_price(0.4523, direction="up") == 0.4523


def test_dollar_boundary() -> None:
    assert quantize_price(1.0, direction="up") == 1.0
    assert quantize_price(1.0, direction="down") == 1.0
    assert quantize_price(0.99997, direction="up") == 1.0
    assert quantize_price(0.99997, direction="down") == 0.9999


@pytest.mark.parametrize("direction", ["up", "down"])
def test_result_is_always_on_the_penny_grid_at_or_above_a_dollar(
    direction: str,
) -> None:
    for price in (37.3297, 105.4975, 12.345678, 8.005):
        q = quantize_price(price, direction=direction)  # type: ignore[arg-type]
        assert round(q * 100) == pytest.approx(q * 100)
