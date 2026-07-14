"""DISPLACEMENT — evict the weakest qualifying position to fund a
HIGH-conviction entry that KS-7 would otherwise reject for lack of cash.

Default OFF (`ExecutionConfig.displacement_enabled`). When enabled, the
agent loop invokes `DisplacementEngine.attempt()` after — and only after
— the sizing engine rejected a proposal with `ks7_cash_reserve` (cash
cannot fund one share above the reserve floor). The engine then:

  1. gates on conviction (`displacement_min_conviction`) and the hard
     ONE-per-trading-day cap (code constant, not configurable);
  2. scans open BROKER positions for qualifying victims:
       - trail NOT armed (`exit_policy_state.trail_engaged` false or no
         row — a position that never reached activation);
       - held >= `MIN_HOLD_TRADING_DAYS` ET trading days;
       - not itself entered today (ET);
       - unrealized gain below `displacement_victim_max_gain_pct` — the
         GRADUATED tier: proposals with conviction >=
         `GAIN_CUTOFF_EXEMPT_CONVICTION` (9) ignore this cutoff, but
         armed positions stay untouchable at EVERY conviction;
  3. picks the weakest (lowest unrealized %, symbol tie-break);
  4. PRE-SIZES the new entry with `displacement_credit =
     displacement_proceeds_haircut × victim market value` — if even the
     credited size cannot fund one share, the displacement aborts BEFORE
     any order is submitted (never evict a position we cannot replace);
  5. submits the victim's liquidation (market sell, then best-effort
     cancel of its live GTC protective legs — the SAME order path the
     thesis-review `close` decision uses) and journals the sale with a
     `displacement:` notes prefix carrying victim, new symbol, and both
     convictions (the audit trail AND the daily-cap persistence);
  6. returns the pre-computed `SizingAccepted` so the loop submits the
     buy through the unchanged OrderSubmitter path.

SELL-BEFORE-BUY GUARANTEE: the `SizingAccepted` is returned only after
the broker ACCEPTED the victim's sell. If the sell submission fails for
any reason, `attempt()` returns None and the loop's normal
`ks7_cash_reserve` rejection stands (journal reason unchanged) — the buy
can never submit without the sell having been accepted first.

Deterministic only: no LLM calls anywhere in this path.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from broker.protocol import (
    AccountSnapshot,
    BrokerAdapter,
    OpenOrder,
    OrderRequest,
    Position,
    Quote,
)
from config.calendar import ET, is_market_open_day
from config.loader import ExecutionConfig
from execution.sizing import SizingAccepted, SizingEngine
from journal.exit_policy import get_exit_policy_state
from journal.models import OrderRow
from journal.repo import connect, get_orders_for_execution, insert_order
from observability.log_port import get_logger
from proposal.schemas import TradeProposal

log = get_logger(__name__)

# Minimum holding age (ET trading days since entry) before a position may
# be evicted. Hard-coded per spec — displacement is for stale dead weight,
# not for churning fresh entries.
MIN_HOLD_TRADING_DAYS = 5

# GRADUATED eligibility: conviction at/above which the victim gain cutoff
# (`displacement_victim_max_gain_pct`) is waived. 9s are near-extinct
# under the review doctrine, so when one appears it may displace even an
# unarmed drifter sitting on a modest gain; base-tier proposals only
# clear sub-cutoff dead weight. Armed positions are untouchable at every
# conviction.
GAIN_CUTOFF_EXEMPT_CONVICTION = 9

# Journal notes prefix on the victim's liquidation order row. Doubles as
# the ONE-per-trading-day cap's persistence: the cap query scans for the
# newest row with this prefix and compares its ET submission date.
DISPLACEMENT_NOTES_PREFIX = "displacement:"


@dataclass(frozen=True)
class DisplacementVictim:
    """A qualifying eviction candidate, resolved from broker + journal."""

    execution_id: int
    conviction: int | None  # the victim's original proposal conviction
    symbol: str
    qty: int
    market_value: float
    unrealized_pct: float  # percent gain/loss vs cost basis


def _ensure_aware(d: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes for TIMESTAMP columns; journal
    timestamps are UTC by convention (mirrors app.loop / coordinator)."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.UTC)
    return d


def trading_days_held(entry_date: dt.date, today: dt.date) -> int:
    """ET trading days elapsed since entry: market-open days `d` with
    `entry_date < d <= today`. Entered today (or a future date) → 0."""
    if entry_date >= today:
        return 0
    days = 0
    d = entry_date + dt.timedelta(days=1)
    while d <= today:
        if is_market_open_day(d):
            days += 1
        d += dt.timedelta(days=1)
    return days


def displaced_today(db_path: str, *, now: dt.datetime) -> bool:
    """Hard cap: has a displacement already run this ET trading day?

    Reads the NEWEST `displacement:`-prefixed order row — rows are only
    written after a broker-accepted victim sale, so the newest row's ET
    submission date is the last displacement's trading day.
    """
    today_et = now.astimezone(ET).date()
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT submitted_at FROM orders WHERE notes LIKE ? "
            "ORDER BY id DESC LIMIT 1",
            (f"{DISPLACEMENT_NOTES_PREFIX}%",),
        ).fetchone()
    if row is None:
        return False
    sub = row["submitted_at"]
    if isinstance(sub, bytes):
        sub = sub.decode()
    if isinstance(sub, str):
        sub = dt.datetime.fromisoformat(sub)
    return _ensure_aware(sub).astimezone(ET).date() == today_et


class DisplacementEngine:
    """Deterministic victim selection + liquidation. One per call site;
    stateless — the journal (`displacement:` order rows) is the only
    persistence, so the daily cap survives restarts."""

    def __init__(
        self,
        *,
        db_path: str,
        broker: BrokerAdapter,
        now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self._db = db_path
        self._broker = broker
        self._now_fn = now_fn

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def attempt(
        self,
        *,
        proposal: TradeProposal,
        proposal_id: int,
        account: AccountSnapshot,
        open_positions: Sequence[Position],
        quote: Quote,
        cfg: ExecutionConfig,
        sizer: SizingEngine,
        pending_buys: list[OpenOrder] | None = None,
        pending_entry_spend: float = 0.0,
    ) -> SizingAccepted | None:
        """Run the full displacement decision for a ks7-rejected proposal.

        Returns the `SizingAccepted` for the new entry (victim sale
        already accepted at the broker) or None — the caller's normal
        rejection then stands, journal reason unchanged.
        """
        if not cfg.displacement_enabled:
            return None
        if proposal.conviction < cfg.displacement_min_conviction:
            return None
        now = self._now_fn()
        if displaced_today(self._db, now=now):
            log.info(
                "execution.displacement.daily_cap",
                extra={
                    "event": "execution.displacement.daily_cap",
                    "proposal_id": proposal_id,
                    "symbol": proposal.symbol,
                },
            )
            return None

        victim = self.find_victim(
            open_positions,
            new_symbol=proposal.symbol,
            conviction=proposal.conviction,
            cfg=cfg,
        )
        if victim is None:
            log.info(
                "execution.displacement.no_qualifying_victim",
                extra={
                    "event": "execution.displacement.no_qualifying_victim",
                    "proposal_id": proposal_id,
                    "symbol": proposal.symbol,
                    "conviction": proposal.conviction,
                },
            )
            return None

        # PRE-SIZE before touching the broker: if even the haircut credit
        # cannot fund one share (or another gate now trips), abort with
        # ZERO orders submitted — never evict a position the proceeds of
        # which cannot fund the replacement. All other gates (KS-4/5/6,
        # tier rates) run unchanged inside the sizer.
        credit = cfg.displacement_proceeds_haircut * victim.market_value
        sizing = sizer.size(
            proposal,
            account,
            list(open_positions),
            quote,
            cfg,
            pending_buys=pending_buys,
            pending_entry_spend=pending_entry_spend,
            displacement_credit=credit,
        )
        if not isinstance(sizing, SizingAccepted):
            log.info(
                "execution.displacement.resize_rejected",
                extra={
                    "event": "execution.displacement.resize_rejected",
                    "proposal_id": proposal_id,
                    "symbol": proposal.symbol,
                    "victim_symbol": victim.symbol,
                    "credit": credit,
                },
            )
            return None

        # SELL FIRST: the buy must never submit unless the victim sale
        # was accepted by the broker. A failed sale aborts displacement
        # entirely — the caller journals the original ks7 rejection.
        if not self.liquidate_victim(
            victim,
            new_symbol=proposal.symbol,
            new_conviction=proposal.conviction,
        ):
            return None

        log.info(
            "execution.displacement",
            extra={
                "event": "execution.displacement",
                "proposal_id": proposal_id,
                "symbol": proposal.symbol,
                "conviction": proposal.conviction,
                "victim_symbol": victim.symbol,
                "victim_execution_id": victim.execution_id,
                "victim_conviction": victim.conviction,
                "victim_unrealized_pct": victim.unrealized_pct,
                "victim_market_value": victim.market_value,
                "proceeds_credit": credit,
                "realized_dollar_size": sizing.realized_dollar_size,
                "qty": sizing.qty,
            },
        )
        return sizing

    # ------------------------------------------------------------------
    # Victim qualification + selection
    # ------------------------------------------------------------------

    def find_victim(
        self,
        open_positions: Sequence[Position],
        *,
        new_symbol: str,
        conviction: int,
        cfg: ExecutionConfig,
    ) -> DisplacementVictim | None:
        """Deterministic victim pick: qualify, then lowest unrealized %
        wins (symbol tie-break for stable ordering)."""
        today_et = self._now_fn().astimezone(ET).date()
        ignore_gain_cutoff = conviction >= GAIN_CUTOFF_EXEMPT_CONVICTION
        candidates: list[DisplacementVictim] = []
        for pos in open_positions:
            if pos.qty <= 0 or pos.symbol == new_symbol:
                continue
            lineage = self._latest_accepted_execution(pos.symbol)
            if lineage is None:
                # No journal lineage (operator-manual position, foreign
                # box state): unauditable age/armed status → not evictable.
                continue
            execution_id, victim_conviction = lineage
            state = get_exit_policy_state(self._db, execution_id=execution_id)
            if state is not None and state.trail_engaged:
                continue  # trail armed — untouchable at every conviction
            entry = next(
                (
                    o
                    for o in get_orders_for_execution(self._db, execution_id)
                    if o.role == "entry"
                ),
                None,
            )
            if entry is None:
                continue
            entry_date_et = (
                _ensure_aware(entry.submitted_at).astimezone(ET).date()
            )
            if entry_date_et == today_et:
                continue  # entered today — never displace a same-day entry
            if trading_days_held(entry_date_et, today_et) < MIN_HOLD_TRADING_DAYS:
                continue
            cost = pos.avg_entry_price * pos.qty
            if cost <= 0:
                continue
            unrealized_pct = pos.unrealized_pnl / cost * 100.0
            if (
                not ignore_gain_cutoff
                and unrealized_pct >= cfg.displacement_victim_max_gain_pct
            ):
                continue
            candidates.append(
                DisplacementVictim(
                    execution_id=execution_id,
                    conviction=victim_conviction,
                    symbol=pos.symbol,
                    qty=pos.qty,
                    market_value=pos.market_value,
                    unrealized_pct=unrealized_pct,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda v: (v.unrealized_pct, v.symbol))

    def _latest_accepted_execution(
        self, symbol: str
    ) -> tuple[int, int | None] | None:
        """Newest accepted execution (and its proposal conviction) for a
        held symbol — the same journal-lineage resolution the boot re-arm
        sweep uses (broker truth says the position is open; the newest
        accepted execution is the entry that opened it)."""
        with connect(self._db) as conn:
            row = conn.execute(
                """
                SELECT e.id AS execution_id, p.conviction AS conviction
                FROM executions e
                JOIN proposals p ON p.id = e.proposal_id
                WHERE e.decision = 'accepted' AND p.symbol = ?
                ORDER BY e.id DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        cv = row["conviction"]
        return int(row["execution_id"]), (int(cv) if cv is not None else None)

    # ------------------------------------------------------------------
    # Victim liquidation (mirrors thesis-review `close` — market sell
    # first, then best-effort cancel of the live GTC protective legs)
    # ------------------------------------------------------------------

    def liquidate_victim(
        self,
        victim: DisplacementVictim,
        *,
        new_symbol: str,
        new_conviction: int,
    ) -> bool:
        """Submit the victim's market sell; True only after broker ack.

        Same order path as `ThesisReviewCoordinator._apply_close`: plain
        market sell (queues to the open when pre-market), journaled
        against the victim's execution, then best-effort cancellation of
        its live protective legs. Qty is the broker position's qty (full
        liquidation of what is actually held).
        """
        req = OrderRequest(
            symbol=victim.symbol,
            side="sell",
            qty=victim.qty,
            order_type="market",
            tif="day",
            client_order_id=f"displace-close-exec-{victim.execution_id}",
        )
        try:
            submitted = self._broker.submit_order(req)
        except Exception as e:  # noqa: BLE001 — abort displacement entirely
            log.error(
                "execution.displacement.sell_failed",
                extra={
                    "event": "execution.displacement.sell_failed",
                    "victim_symbol": victim.symbol,
                    "victim_execution_id": victim.execution_id,
                    "new_symbol": new_symbol,
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
            return False

        insert_order(
            self._db,
            OrderRow(
                execution_id=victim.execution_id,
                role="displacement_close",
                symbol=victim.symbol,
                side="sell",
                order_type="market",
                qty=victim.qty,
                tif="day",
                broker_order_id=submitted.broker_order_id,
                submitted_at=submitted.submitted_at,
                final_status=submitted.status,
                notes=(
                    f"{DISPLACEMENT_NOTES_PREFIX} evicted {victim.symbol} "
                    f"(execution {victim.execution_id}, "
                    f"unrealized {victim.unrealized_pct:+.2f}%, "
                    f"conviction {victim.conviction}) "
                    f"to fund {new_symbol} (conviction {new_conviction})"
                ),
            ),
        )

        # Cancel the live GTC protective legs, best effort (mirrors the
        # thesis close). Failures are not fatal — the position is closing
        # and a lingering leg on a zero-qty position cannot fill.
        for o in get_orders_for_execution(self._db, victim.execution_id):
            if (
                o.role in ("stop", "take_profit", "trailing_stop")
                and o.final_status is None
            ):
                try:
                    self._broker.cancel_order(o.broker_order_id)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "execution.displacement.cancel_failed",
                        extra={
                            "event": "execution.displacement.cancel_failed",
                            "victim_execution_id": victim.execution_id,
                            "role": o.role,
                            "broker_order_id": o.broker_order_id,
                            "error": str(e),
                        },
                    )
        return True


__all__ = [
    "DISPLACEMENT_NOTES_PREFIX",
    "GAIN_CUTOFF_EXEMPT_CONVICTION",
    "MIN_HOLD_TRADING_DAYS",
    "DisplacementEngine",
    "DisplacementVictim",
    "displaced_today",
    "trading_days_held",
]
