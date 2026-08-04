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
  5. submits the victim's liquidation (confirmed cancel of its live GTC
     protective legs FIRST, then the market sell — the SAME
     protection-preserving order path the thesis-review `close`
     decision uses) and journals the sale with a
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
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

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
from execution.protection import cancel_protective_legs, restore_protection
from execution.sizing import SizingAccepted, SizingEngine
from journal.exit_policy import get_exit_policy_state
from journal.models import OrderRow
from journal.repo import (
    connect,
    get_latest_chopping_block,
    get_orders_for_execution,
    insert_order,
)
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
    days_held: int  # ET trading days held since entry
    # True when this victim was selected via the young-victim AGE OVERRIDE
    # (younger than MIN_HOLD_TRADING_DAYS, admissible only from the
    # chopping block for a conviction >= displacement_young_victim_min_conviction).
    via_override: bool = False


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


def _block_is_fresh(block_date: str, today_et: dt.date) -> bool:
    """True when the chopping block dated `block_date` (ET) is from the
    CURRENT or the immediately PREVIOUS trading session relative to
    `today_et` — the freshness the AGE OVERRIDE requires. A block older
    than one trading session (a weekend that spans a holiday, or one or
    more failed sweeps) is stale: `trading_days_held` counts >= 2 market
    days between it and today. A block dated today (or, defensively, in
    the future) is always fresh."""
    try:
        bd = dt.date.fromisoformat(block_date)
    except ValueError:
        return False
    if bd >= today_et:
        return True
    return trading_days_held(bd, today_et) <= 1


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
                "victim_days_held": victim.days_held,
                "override": victim.via_override,
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
        """Pick the victim to evict.

        With no chopping block (feature off / no block persisted yet), this
        is the original deterministic pick: qualify (unarmed, aged >= 5
        trading days, not entered today, sub-cutoff) then lowest unrealized
        % wins (symbol tie-break). When a block EXISTS and
        `displacement_use_chopping_block` is on, the SAME eligibility gates
        apply but the ORDER becomes the block's thesis-sanctioned rank
        (rank 1 = most expendable first). When no age-eligible victim
        exists at all, the AGE OVERRIDE may reach a younger victim — but
        only one on the block, and only for a proposal at/above
        `displacement_young_victim_min_conviction`.
        """
        today_et = self._now_fn().astimezone(ET).date()
        ignore_gain_cutoff = conviction >= GAIN_CUTOFF_EXEMPT_CONVICTION
        block = self._load_chopping_block(cfg)
        block_ranks: dict[int, int] | None = None
        block_date: str | None = None
        if block is not None:
            block_ranks, block_date = block

        qualified: list[DisplacementVictim] = []
        for pos in open_positions:
            victim = self._qualify_victim(
                pos,
                new_symbol=new_symbol,
                today_et=today_et,
                ignore_gain_cutoff=ignore_gain_cutoff,
                cfg=cfg,
            )
            if victim is not None:
                qualified.append(victim)
        if not qualified:
            return None

        aged = [v for v in qualified if v.days_held >= MIN_HOLD_TRADING_DAYS]

        if block_ranks is not None:
            # Block present: iterate age-eligible victims in block-rank
            # order (already revalidated open/unarmed/qty>0 by qualify).
            aged_in_block = [v for v in aged if v.execution_id in block_ranks]
            if aged_in_block:
                return min(
                    aged_in_block, key=lambda v: block_ranks[v.execution_id]
                )
            if aged:
                # Block exists but none of the aged victims are on it
                # (entered after the sweep, or the sweep didn't rank them):
                # fall back to the deterministic weakest-first pick rather
                # than strand a legitimate displacement.
                return min(aged, key=lambda v: (v.unrealized_pct, v.symbol))
            # No age-eligible victim at all → AGE OVERRIDE, chopping-block
            # only: evict the weakest-rank YOUNG victim on the block when
            # the proposal clears the override conviction gate.
            min_conv = cfg.displacement_young_victim_min_conviction
            if min_conv and conviction >= min_conv:
                # FRESHNESS GATE: the override evicts a position younger than
                # the normal dead-weight rule, so it demands a CURRENT thesis
                # sanction. A block older than the previous trading session
                # (a weekend/holiday/failed-sweep gap) is stale — refuse the
                # override and fall through to no-young-victims. Ordinary
                # rank-ordering above may still use a latest-but-stale block
                # because those victims are fully re-qualified live.
                if block_date is None or not _block_is_fresh(
                    block_date, today_et
                ):
                    log.info(
                        "execution.displacement.stale_block_override_skipped",
                        extra={
                            "event": "execution.displacement.stale_block_override_skipped",
                            "new_symbol": new_symbol,
                            "conviction": conviction,
                            "block_date": block_date,
                            "today_et": today_et.isoformat(),
                        },
                    )
                    return None
                young_in_block = [
                    v
                    for v in qualified
                    if v.days_held < MIN_HOLD_TRADING_DAYS
                    and v.execution_id in block_ranks
                ]
                if young_in_block:
                    chosen = min(
                        young_in_block,
                        key=lambda v: block_ranks[v.execution_id],
                    )
                    return replace(chosen, via_override=True)
            return None

        # No block (or feature off): original deterministic behavior —
        # weakest age-eligible victim, byte-identical to pre-feature.
        if not aged:
            return None
        return min(aged, key=lambda v: (v.unrealized_pct, v.symbol))

    def _qualify_victim(
        self,
        pos: Position,
        *,
        new_symbol: str,
        today_et: dt.date,
        ignore_gain_cutoff: bool,
        cfg: ExecutionConfig,
    ) -> DisplacementVictim | None:
        """Apply every eviction gate EXCEPT the age gate (the caller decides
        whether the normal >= 5-trading-day rule or the age override
        applies). Returns a victim carrying its `days_held`, or None when
        the position can never be a victim (wrong symbol, qty<=0, no
        lineage, armed, entered today, no entry order, zero cost, or a gain
        at/above the cutoff at the base tier)."""
        if pos.qty <= 0 or pos.symbol == new_symbol:
            return None
        lineage = self._latest_accepted_execution(pos.symbol)
        if lineage is None:
            # No journal lineage (operator-manual position, foreign box
            # state): unauditable age/armed status → not evictable.
            return None
        execution_id, victim_conviction = lineage
        state = get_exit_policy_state(self._db, execution_id=execution_id)
        if state is not None and state.trail_engaged:
            return None  # trail armed — untouchable at every conviction
        entry = next(
            (
                o
                for o in get_orders_for_execution(self._db, execution_id)
                if o.role == "entry"
            ),
            None,
        )
        if entry is None:
            return None
        entry_date_et = _ensure_aware(entry.submitted_at).astimezone(ET).date()
        if entry_date_et == today_et:
            return None  # entered today — never displace a same-day entry
        cost = pos.avg_entry_price * pos.qty
        if cost <= 0:
            return None
        unrealized_pct = pos.unrealized_pnl / cost * 100.0
        if (
            not ignore_gain_cutoff
            and unrealized_pct >= cfg.displacement_victim_max_gain_pct
        ):
            return None
        return DisplacementVictim(
            execution_id=execution_id,
            conviction=victim_conviction,
            symbol=pos.symbol,
            qty=pos.qty,
            market_value=pos.market_value,
            unrealized_pct=unrealized_pct,
            days_held=trading_days_held(entry_date_et, today_et),
        )

    def _load_chopping_block(
        self, cfg: ExecutionConfig
    ) -> tuple[dict[int, int], str] | None:
        """`(execution_id -> rank, block_date)` for the most recent chopping
        block, or None when the feature is off / no block has been
        persisted. A missing `chopping_block` table (a DB migrated before
        this feature) is treated the same as an empty block — displacement
        then falls back to the deterministic ranking. `block_date` is the
        ET date the block was written; the AGE OVERRIDE uses it to require a
        fresh thesis sanction (see `_block_is_fresh`)."""
        if not cfg.displacement_use_chopping_block:
            return None
        try:
            rows = get_latest_chopping_block(self._db)
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        return {r.execution_id: r.rank for r in rows}, rows[0].block_date

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
    # Victim liquidation (mirrors thesis-review `close` — confirmed
    # cancel of the live GTC protective legs first, THEN the market
    # sell; failure re-places the victim's protection)
    # ------------------------------------------------------------------

    def liquidate_victim(
        self,
        victim: DisplacementVictim,
        *,
        new_symbol: str,
        new_conviction: int,
    ) -> bool:
        """Cancel the victim's live protective legs (confirmed), THEN
        submit its market sell; True only after broker ack of the sell.

        Same protection-preserving sequence as the thesis-review `close`
        (incident VRDN 2026-07-14..17: a sell submitted while GTC legs
        hold the shares is rejected with Alpaca 40310000 `insufficient
        qty available`). Any failure aborts the displacement with the
        victim's protection re-placed — the victim must never end LESS
        protected because a displacement half-ran. Qty is the broker
        position's qty (full liquidation of what is actually held).
        """
        outcome = cancel_protective_legs(
            self._db, self._broker, victim.execution_id
        )
        if outcome.any_filled:
            # A protective leg filled mid-flight — the victim is already
            # exiting on its own; abort the displacement cleanly.
            log.info(
                "execution.displacement.victim_leg_filled",
                extra={
                    "event": "execution.displacement.victim_leg_filled",
                    "victim_symbol": victim.symbol,
                    "victim_execution_id": victim.execution_id,
                },
            )
            return False
        if not outcome.all_cleared:
            if outcome.cleared:
                restore_protection(
                    self._db,
                    self._broker,
                    victim.execution_id,
                    outcome.cleared,
                    client_order_id=(
                        f"displace-restore-exec-{victim.execution_id}"
                    ),
                    notes="displacement aborted: leg cancel failed; protection restored",
                )
            log.warning(
                "execution.displacement.cancel_failed",
                extra={
                    "event": "execution.displacement.cancel_failed",
                    "victim_execution_id": victim.execution_id,
                    "victim_symbol": victim.symbol,
                    "still_live_broker_order_ids": [
                        o.broker_order_id for o in outcome.failed
                    ],
                },
            )
            return False

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
            restored = restore_protection(
                self._db,
                self._broker,
                victim.execution_id,
                outcome.cleared,
                client_order_id=f"displace-restore-exec-{victim.execution_id}",
                notes="displacement aborted: sell failed; protection restored",
            )
            log.error(
                "execution.displacement.sell_failed",
                extra={
                    "event": "execution.displacement.sell_failed",
                    "victim_symbol": victim.symbol,
                    "victim_execution_id": victim.execution_id,
                    "new_symbol": new_symbol,
                    "protection_restored": restored is not None,
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
                    f"held {victim.days_held}d, "
                    f"conviction {victim.conviction}) "
                    f"to fund {new_symbol} (conviction {new_conviction})"
                    + (" [young-victim age override]" if victim.via_override else "")
                ),
            ),
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
