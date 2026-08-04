"""CALIBRATION memory provider — the analyzer's own realized track
record by conviction band.

Covers: purpose gating (analyze only — None for proposal_review /
thesis_review), None with zero closed trades, the small-sample (N<10)
single-line section, the full deterministic render (band lines +
anti-overfit framing), N=0 band rendering, and rail integration
(assembles under the shared MemoryContextAssembler).
"""

from __future__ import annotations

from typing import Any

from app.memory_calibration import CalibrationMemoryProvider
from app.memory_context import MemoryContextAssembler, MemoryQuery


class _FakeJournal:
    """Duck-typed stand-in for JournalRepo.get_conviction_calibration."""

    def __init__(self, calib: dict[str, Any]) -> None:
        self._calib = calib
        self.calls = 0

    def get_conviction_calibration(self) -> dict[str, Any]:
        self.calls += 1
        return self._calib


def _band(
    band: str,
    n: int,
    win: float | None = None,
    mean: float | None = None,
    median: float | None = None,
) -> dict[str, Any]:
    return {
        "band": band,
        "n": n,
        "win_rate_pct": win,
        "mean_realized_pct": mean,
        "median_realized_pct": median,
    }


def _empty_calib() -> dict[str, Any]:
    return {
        "total_n": 0,
        "bands": [
            _band("2-3", 0),
            _band("4-5", 0),
            _band("6-7", 0),
            _band("8-10", 0),
        ],
    }


def _full_calib() -> dict[str, Any]:
    return {
        "total_n": 42,
        "bands": [
            _band("2-3", 4, 25.0, -8.2, -7.5),
            _band("4-5", 21, 42.9, -1.3, -0.8),
            _band("6-7", 14, 57.1, 3.8, 2.9),
            _band("8-10", 3, 66.7, 11.0, 9.4),
        ],
    }


def _q(purpose: str = "analyze") -> MemoryQuery:
    return MemoryQuery(symbol="ACME", purpose=purpose)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# purpose gating
# ---------------------------------------------------------------------------


def test_none_for_thesis_review() -> None:
    journal = _FakeJournal(_full_calib())
    provider = CalibrationMemoryProvider(journal)
    assert provider(_q("thesis_review")) is None
    assert provider(_q("proposal_review")) is None
    # gated purposes never touch the journal
    assert journal.calls == 0


def test_serves_for_analyze() -> None:
    provider = CalibrationMemoryProvider(_FakeJournal(_full_calib()))
    section = provider(_q("analyze"))
    assert section is not None
    assert section.provider_name == "calibration"


# ---------------------------------------------------------------------------
# empty / small-sample behavior
# ---------------------------------------------------------------------------


def test_none_with_zero_closed_trades() -> None:
    provider = CalibrationMemoryProvider(_FakeJournal(_empty_calib()))
    assert provider(_q()) is None


def test_small_sample_single_line() -> None:
    calib = {
        "total_n": 7,
        "bands": [
            _band("2-3", 0),
            _band("4-5", 3, 33.3, -2.0, -1.5),
            _band("6-7", 4, 50.0, 1.0, 0.5),
            _band("8-10", 0),
        ],
    }
    provider = CalibrationMemoryProvider(_FakeJournal(calib))
    section = provider(_q())
    assert section is not None
    assert "\n" not in section.body
    assert "7 closed trades" in section.body
    assert "too small" in section.body
    # no per-band stats leak into the small-sample line
    assert "cv4-5" not in section.body


# ---------------------------------------------------------------------------
# full render
# ---------------------------------------------------------------------------


def test_full_render_body() -> None:
    provider = CalibrationMemoryProvider(_FakeJournal(_full_calib()))
    section = provider(_q())
    assert section is not None
    assert section.title == "Conviction calibration"
    assert section.body == (
        "Your own realized track record by conviction "
        "(N=42 closed trades — small sample, treat as weak prior): "
        "cv2-3: N=4, win 25.0%, mean -8.2%, median -7.5%. "
        "cv4-5: N=21, win 42.9%, mean -1.3%, median -0.8%. "
        "cv6-7: N=14, win 57.1%, mean +3.8%, median +2.9%. "
        "cv8-10: N=3, win 66.7%, mean +11.0%, median +9.4%. "
        "Calibrate scores against this record; do not let it override "
        "filing-specific evidence."
    )


def test_empty_band_renders_n_zero() -> None:
    calib = {
        "total_n": 12,
        "bands": [
            _band("2-3", 0),
            _band("4-5", 6, 50.0, 0.5, 0.2),
            _band("6-7", 6, 50.0, 1.5, 1.2),
            _band("8-10", 0),
        ],
    }
    provider = CalibrationMemoryProvider(_FakeJournal(calib))
    section = provider(_q())
    assert section is not None
    assert "cv2-3: N=0." in section.body
    assert "cv8-10: N=0." in section.body


def test_render_is_deterministic() -> None:
    provider = CalibrationMemoryProvider(_FakeJournal(_full_calib()))
    assert provider(_q()) == provider(_q())


# ---------------------------------------------------------------------------
# rail integration
# ---------------------------------------------------------------------------


def test_assembles_under_the_shared_rail() -> None:
    assembler = MemoryContextAssembler()
    assembler.register(
        "calibration", CalibrationMemoryProvider(_FakeJournal(_full_calib()))
    )
    out = assembler.assemble(_q("analyze"))
    assert out is not None
    assert out.startswith("## MEMORY: Conviction calibration\n")
    assert assembler.assemble(_q("thesis_review")) is None
