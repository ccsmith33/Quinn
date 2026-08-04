"""CALIBRATION memory provider — the analyzer's own track record.

Serves a compact table of realized outcomes by conviction band (from
`journal.repo.get_conviction_calibration`) so conviction scoring is
grounded in how the system's past scores actually performed. Registered
on the shared `MemoryContextAssembler` rail as "calibration" (gated by
`config.memory.calibration_enabled` at the composition seam).

Scope: `purpose == 'analyze'` ONLY — the filing analyzer is the one
being calibrated; proposal / thesis reviews get None (their job is
evaluating a specific trade, not scoring conviction from scratch).

Small-sample policy (documented choice):
- 0 closed trades  → None (nothing to say; the rail renders no section).
- 1-9 closed trades → a single-line section stating the sample is too
  small for calibration. Surfacing "a record exists but is too thin"
  beats silence: it pre-empts the analyzer inventing a track record.
- 10+ closed trades → the full per-band table, with the anti-overfit
  framing ("small sample, treat as weak prior … do not let it override
  filing-specific evidence") baked into the body text.

Deterministic: the body is a pure render of the repo's fixed-order band
stats — identical journal state yields byte-identical output.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.memory_context import MemoryQuery, MemorySection

_TITLE = "Conviction calibration"
_PROVIDER_NAME = "calibration"
_MIN_SAMPLE = 10


class _CalibrationJournal(Protocol):
    """The one slice of JournalRepo this provider needs."""

    def get_conviction_calibration(self) -> dict[str, Any]: ...


class CalibrationMemoryProvider:
    """`Provider`-shaped callable: `(MemoryQuery) -> MemorySection | None`."""

    def __init__(self, journal: _CalibrationJournal) -> None:
        self._journal = journal

    def __call__(self, query: MemoryQuery) -> MemorySection | None:
        if query.purpose != "analyze":
            return None
        calib = self._journal.get_conviction_calibration()
        total_n = int(calib["total_n"])
        if total_n == 0:
            return None
        if total_n < _MIN_SAMPLE:
            body = (
                f"Only {total_n} closed trades on record — too small a "
                "sample for conviction calibration; rely on "
                "filing-specific evidence."
            )
            return MemorySection(
                title=_TITLE, body=body, provider_name=_PROVIDER_NAME
            )
        parts: list[str] = []
        for band in calib["bands"]:
            if band["n"] == 0:
                parts.append(f"cv{band['band']}: N=0.")
                continue
            parts.append(
                f"cv{band['band']}: N={band['n']}, "
                f"win {band['win_rate_pct']:.1f}%, "
                f"mean {band['mean_realized_pct']:+.1f}%, "
                f"median {band['median_realized_pct']:+.1f}%."
            )
        body = (
            "Your own realized track record by conviction "
            f"(N={total_n} closed trades — small sample, treat as weak "
            "prior): "
            + " ".join(parts)
            + " Calibrate scores against this record; do not let it "
            "override filing-specific evidence."
        )
        return MemorySection(title=_TITLE, body=body, provider_name=_PROVIDER_NAME)


__all__ = ["CalibrationMemoryProvider"]
