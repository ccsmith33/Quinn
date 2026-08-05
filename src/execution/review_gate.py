"""Full-book Opus-review gate (API cost lever) — pure logic.

Business context: Opus review is the expensive half of the per-filing
spend, and it is paid on every proposal at/above
`analyzer.opus_review_conviction_threshold` (5 in prod) — including
proposals arriving when the book is full AND cash sits at the KS-7
reserve floor, where nothing below displacement conviction can be bought
no matter what Opus says.

When both conditions hold, this module raises the effective review bar to
the lowest conviction that could still fund itself by displacing an
existing position. Proposals under the raised bar skip Opus; the agent
loop journals them with `review_skipped_full_book` so they stay in the
history and stay eligible for the watchlist's deferred retry.

STATELESS by design (requirement #5): the pressure snapshot is taken per
proposal at decision time, so the gate reverts the instant a slot frees
or cash returns. There is no latch to reset and nothing to reconcile
after a restart.

Slot math mirrors `SizingEngine.size`'s KS-5 gate and
`ThesisCoordinator._compute_capacity_pressure_block` exactly: used slots
are the UNIQUE symbols across held positions ∪ pending entry BUYs, and
the cap is the equity-tiered `ks5_effective_cap`. Cash math mirrors
KS-7: headroom is broker cash, less the dollars pending entries have
already committed, less the reserve floor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from config.loader import ExecutionConfig
from execution.sizing import ks5_effective_cap
from observability.log_port import get_logger

log = get_logger(__name__)

# The conviction bar to apply when displacement is OFF. With no escape
# hatch, a full book with no cash cannot fund ANY entry, so the bar could
# in principle be 11 (skip everything). We use 8 — the displacement
# feature's own default floor — so that flipping `displacement_enabled`
# on or off does not change which proposals this gate skips. That keeps
# the two features independently reversible, which matters more than the
# marginal saving on cv8-10 proposals (a small fraction of the volume).
NO_DISPLACEMENT_FLOOR = 8


@dataclass(frozen=True)
class SlotUsage:
    """KS-5 slot accounting for one decision.

    The single definition of "how full is the book", shared by the
    Opus-review gate and the thesis coordinator's capacity-pressure
    sweep. `SizingEngine.size` performs the same computation inline as
    the authoritative pre-trade gate; the two agree because both count
    UNIQUE symbols across held ∪ pending-entry BUYs against
    `ks5_effective_cap`. (Carry-forward of the F4 note in
    `thesis_coordinator`: before this type the two read sites assembled
    their inputs independently and agreed only by construction.)
    """

    cap: int
    held_symbols: frozenset[str]
    pending_symbols: frozenset[str]

    @property
    def used(self) -> int:
        return len(self.held_symbols | self.pending_symbols)

    @property
    def open_slots(self) -> int:
        return self.cap - self.used

    @property
    def pending_only(self) -> frozenset[str]:
        """Pending entries on names not already held — the count that is
        meaningful to an operator reading a capacity log line."""
        return self.pending_symbols - self.held_symbols


def slot_usage(
    *,
    equity: float,
    positions: Iterable[Any],
    pending_entries: Iterable[Any],
    cfg: ExecutionConfig,
) -> SlotUsage:
    """Pure slot math. Callers own their own broker I/O and error
    handling; only the accounting lives here.

    `positions` must already be filtered to live holdings (`qty != 0`)
    and `pending_entries` to entry BUYs — the two call sites have
    different failure semantics for those reads, so the filtering stays
    with the fetch.
    """
    return SlotUsage(
        cap=ks5_effective_cap(equity, cfg),
        held_symbols=frozenset(p.symbol for p in positions),
        pending_symbols=frozenset(o.symbol for o in pending_entries),
    )


@dataclass(frozen=True)
class BookPressure:
    """Decision-time capacity snapshot.

    `open_slots` may be negative if the broker holds more names than the
    current effective cap allows (an equity drop can lower a tiered cap
    under the existing book); callers treat any value <= 0 as full.
    `cash_headroom` is the dollars deployable without breaching KS-7.
    """

    open_slots: int
    cash_headroom: float


def displacement_viable_conviction(cfg: ExecutionConfig) -> int:
    """Lowest conviction that could still buy against a full, cash-dry
    book — i.e. the lowest conviction displacement will act on.

    `displacement.attempt` hard-gates on `displacement_min_conviction`;
    `displacement_young_victim_min_conviction` only relaxes the victim
    AGE requirement above its own threshold and never lowers that gate.
    We take the lower of the two regardless, so an operator who sets the
    young-victim knob below the base floor cannot cause a proposal that
    displacement would actually have acted on to be skipped here.
    """
    if not getattr(cfg, "displacement_enabled", False):
        return NO_DISPLACEMENT_FLOOR
    floor = int(getattr(cfg, "displacement_min_conviction", NO_DISPLACEMENT_FLOOR))
    young = int(getattr(cfg, "displacement_young_victim_min_conviction", 0))
    return min(floor, young) if young else floor


def compute_book_pressure(
    *,
    broker: Any,
    cfg: ExecutionConfig,
    pending_entry_spend_fn: Callable[[list[Any]], float],
) -> BookPressure | None:
    """Snapshot capacity + cash from broker truth.

    Returns None when the broker cannot be read. None means "gate does
    not fire" at every call site — a broker outage must not start
    silencing Opus.
    """
    try:
        account = broker.get_account()
        positions = [p for p in broker.get_positions() if p.qty != 0]
    except Exception as e:  # noqa: BLE001 — outage → gate off, never blocking
        log.warning(
            "review_gate.broker_snapshot_failed",
            extra={
                "event": "review_gate.broker_snapshot_failed",
                "error": str(e),
            },
        )
        return None

    # Pending entry BUYs consume a slot and commit cash the broker has not
    # debited yet (incident 2026-05-07 for slots, hotfix 2026-07-08 for
    # dollars). Absent on pre-WS1 adapters and test stubs.
    pending_entries: list[Any] = []
    get_open_orders = getattr(broker, "get_open_orders", None)
    if get_open_orders is not None:
        try:
            pending_entries = [o for o in get_open_orders() if o.is_entry]
        except Exception as e:  # noqa: BLE001 — positions are the hard input
            log.warning(
                "review_gate.pending_orders_failed",
                extra={
                    "event": "review_gate.pending_orders_failed",
                    "error": str(e),
                },
            )
            pending_entries = []

    usage = slot_usage(
        equity=account.equity,
        positions=positions,
        pending_entries=pending_entries,
        cfg=cfg,
    )

    try:
        committed = float(pending_entry_spend_fn(pending_entries))
    except Exception as e:  # noqa: BLE001 — unpriced orders → assume nothing
        log.warning(
            "review_gate.pending_spend_failed",
            extra={
                "event": "review_gate.pending_spend_failed",
                "error": str(e),
            },
        )
        committed = 0.0

    reserve_floor = cfg.ks7_cash_reserve_pct * account.equity
    return BookPressure(
        open_slots=usage.open_slots,
        cash_headroom=account.cash - committed - reserve_floor,
    )


def effective_opus_review_threshold(
    *,
    configured_threshold: int,
    cfg: ExecutionConfig,
    pressure: BookPressure | None,
    gate_enabled: bool,
    price_floor_usd: float,
) -> int:
    """The conviction bar Opus review must clear for THIS decision.

    Returns `configured_threshold` unchanged unless the gate is on AND
    the book is full AND KS-7 headroom cannot fund even the cheapest
    legal share. `price_floor_usd` is the system-wide minimum tradeable
    share price, so headroom below it means no entry in the universe is
    fundable — a symbol-independent lower bound that needs no quote.
    """
    if not gate_enabled or pressure is None:
        return configured_threshold
    if pressure.open_slots > 0:
        return configured_threshold
    if pressure.cash_headroom >= price_floor_usd:
        return configured_threshold
    return max(configured_threshold, displacement_viable_conviction(cfg))


def resolve_review_threshold(
    *,
    config: Any,
    broker: Any,
    pending_entry_spend_fn: Callable[[list[Any]], float],
    price_floor_fallback: float = 5.00,
) -> int:
    """The single seam both the analyzer and the agent loop ask for the bar.

    Two callers must agree on this number: `SonnetAnalyzer` decides
    whether to pay for Opus, and `AgentLoop._execute` decides whether a
    missing review means `pending_capacity` (deferred, Feature C will
    retry) or `review_skipped_full_book` (terminal, watchlist-eligible).
    A divergence writes the wrong reject_reason — never a wrong trade —
    but it would strand proposals in the wrong queue, so both go through
    here rather than reimplementing the read.

    Returns the configured threshold on any missing config, disabled
    gate, or error: the failure direction is always "review anyway".
    """
    analyzer_cfg = getattr(config, "analyzer", None)
    execution_cfg = getattr(config, "execution", None)
    configured = int(getattr(analyzer_cfg, "opus_review_conviction_threshold", 0))
    gate_enabled = bool(getattr(analyzer_cfg, "opus_review_full_book_gate", False))
    if not gate_enabled or execution_cfg is None:
        return configured

    try:
        pressure = compute_book_pressure(
            broker=broker,
            cfg=execution_cfg,
            pending_entry_spend_fn=pending_entry_spend_fn,
        )
        return effective_opus_review_threshold(
            configured_threshold=configured,
            cfg=execution_cfg,
            pressure=pressure,
            gate_enabled=True,
            price_floor_usd=float(
                getattr(execution_cfg, "price_floor_usd", None)
                or price_floor_fallback
            ),
        )
    except Exception as e:  # noqa: BLE001 — fail toward reviewing
        log.warning(
            "review_gate.resolve_failed",
            extra={"event": "review_gate.resolve_failed", "error": str(e)},
        )
        return configured


__all__ = [
    "NO_DISPLACEMENT_FLOOR",
    "BookPressure",
    "compute_book_pressure",
    "SlotUsage",
    "displacement_viable_conviction",
    "effective_opus_review_threshold",
    "resolve_review_threshold",
    "slot_usage",
]
