"""Integration (task #4, W2-L3) — one `done_for_day` semantics.

WS2's thesis-coordinator fallback and WS3's converter heal gate each
classify broker order statuses independently. Review W2-L3 caught the
divergence: the coordinator called `done_for_day` dead (re-place
protection) while the converter called it healable (live protection).
An Alpaca GTC order parked `done_for_day` resumes next session — it IS
standing protection, and re-placing over it races a second full-qty
sell against it. Ruling: done_for_day is LIVE everywhere. This test
pins both sets so the next divergence fails loudly.
"""

from __future__ import annotations

from app.thesis_coordinator import _DEAD_ORDER_STATUSES
from execution.pdt_transition import _HEALABLE_ORDER_STATUSES


def test_done_for_day_is_live_in_both_classifiers() -> None:
    assert "done_for_day" not in _DEAD_ORDER_STATUSES
    assert "done_for_day" in _HEALABLE_ORDER_STATUSES


def test_dead_and_healable_sets_do_not_overlap() -> None:
    """A status can't be both 'will never fill' (re-place protection)
    and 'is/was valid protection' (heal). `replaced` and `filled`-class
    statuses live on exactly one side; overlap means one module is
    misclassifying."""
    overlap = _DEAD_ORDER_STATUSES & _HEALABLE_ORDER_STATUSES
    assert not overlap, f"divergent status semantics: {sorted(overlap)}"
