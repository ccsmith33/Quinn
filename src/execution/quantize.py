"""Broker-bound price quantization (hotfix 2026-08-11, ATRC 42210000).

Alpaca rejects any order whose price is off the venue's minimum
increment grid (error 42210000 — "sub-penny increment does not fulfill
minimum pricing criteria"): orders priced >= $1.00 must land on penny
($0.01) increments; orders priced under $1.00 may use $0.0001
increments. Entry FILLS, by contrast, routinely land at sub-penny
prices (ATRC avg fill 37.3297) — so every order price DERIVED from a
fill (breakeven floors, trail math, LLM-suggested levels) must be
quantized before it reaches the broker. This module is the single seam.

Two rounding directions, chosen to preserve invariants:

- ``direction="up"`` — breakeven-floor components and raise-only sell
  LIMITs (TP). The floor invariant is ``stop >= entry``; rounding UP to
  the next increment keeps it (rounding down would break it by a hair).
  A sell limit a fraction higher likewise preserves the raise-only and
  R:R-floor checks and is always broker-valid.
- ``direction="down"`` — trail-derived and protective sell-stop
  components. A protective sell-stop a fraction LOWER is semantically
  identical and always broker-valid.

Sub-$1 prices quantize to 4dp (Alpaca's sub-dollar rule). Quinn's
universe price floor (``execution.price_floor_usd``, default $5.00)
should keep every symbol out of that branch, but the helper handles it
rather than assume the config never changes.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal

_PENNY = Decimal("0.01")
_SUB_DOLLAR = Decimal("0.0001")


def quantize_price(price: float, *, direction: Literal["up", "down"]) -> float:
    """Quantize a broker-bound order price to a valid Alpaca increment.

    Float-noise guard: the input is first rounded to the nearest 1e-6 —
    two orders of magnitude finer than the finest venue increment — so
    binary-float noise from reconstructed prices (33.48 arriving as
    33.479999999999996 out of ``entry − risk``) can't drag a directional
    round across a whole increment. Genuine price content never lives at
    the 1e-6 scale, and ``Decimal(str(...))`` then sees the float's
    shortest repr, so quantizing an already-clean price is the identity
    in both directions.
    """
    d = Decimal(str(round(price, 6)))
    increment = _PENNY if d >= 1 else _SUB_DOLLAR
    rounding = ROUND_CEILING if direction == "up" else ROUND_FLOOR
    return float(d.quantize(increment, rounding=rounding))


__all__ = ["quantize_price"]
