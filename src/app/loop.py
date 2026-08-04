"""S5.6 — agent loop coordinator.

Owns:
- the asyncio consumer task that pulls `FilingRow`s from the ingestion
  queue and walks each through prefilter → analyzer (→ Opus) →
  validator → sizer → submitter (→ journal);
- the BOOTING crash-recovery scan (AC-7);
- the SIGTERM-aware shutdown sequence (AC-6);
- the kill-switch defense-in-depth WARN log (AC-5 step 5);
- per-iteration correlation_id binding (S8.1 hook).

The state machine in `app.state` guards every transition. A single
consumer task processes one filing at a time per architecture §10.5
(D-034). The reconciler runs as a sibling task started during BOOTING.

Carry-forward S5.5 D-1 (medium): on restart, before invoking the
analyzer, look up `proposals.decision_id` for the `(filing_id,
sonnet_model_id, prompt_version)` triple — if present, skip the LLM
call to avoid re-spending. The Opus retry path is similarly guarded
by checking `proposal_reviews.proposal_id`.

Carry-forward D-038 (medium): the universe is loaded once at boot via
`Universe.load_latest`; if `_load_members` returned partial state the
loader retries once with a 50ms sleep before failing. That logic lives
in S2.3's `Universe`, not here — this module just constructs it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
import uuid
from typing import Any

from broker.alpaca import BrokerUnavailable
from broker.protocol import BrokerRejected, OpenOrder
from execution.orders import AcceptedProposal, SubmissionAccepted, SubmissionFailed
from execution.sizing import SizingAccepted, SizingRejected
from execution.validator import Accepted, Rejected
from journal.models import ExecutionRow, FilingRow
from journal.repo import (
    connect,
    get_proposal_by_decision_id,
    get_proposal_by_id,
    get_proposal_review_by_proposal_id,
)
from killswitch.api import KillSwitchUninitialized
from observability.log_port import get_logger
from prompts.loader import AnalyzerContext

from .composition import AgentComponents
from .signals import install_shutdown_signal_handlers
from .state import AgentState, Transitions

log = get_logger(__name__)

# Bound on the BOOTING crash-recovery scan (AC-7). Above this, we let
# the submissions reconciler (S3.4) backstop further recovery rather
# than holding the loop in BOOTING for hours.
_RECOVERY_SCAN_MAX_FILINGS = 500

# Review A2 — broker order states in which an order can no longer fill.
# A `prop-{id}-entry` in one of these states AND with zero filled_qty is
# NOT a live orphan and must not be adopted as a phantom position. The
# zero-fill condition is load-bearing: canceled/expired orders can carry
# a PARTIAL fill, which is real unprotected exposure and MUST still be
# adopted. Everything else (accepted/new/pending_*/partially_filled/
# filled/replaced/held/...) is live or executed and is adopted as before.
_TERMINAL_DEAD_ORDER_STATUSES = frozenset({"rejected", "canceled", "expired"})


class AgentLoop:
    """The long-running asyncio coordinator. One per agent process."""

    def __init__(
        self,
        *,
        components: AgentComponents,
        config: Any | None = None,
        shutdown_grace_seconds: float = 60.0,
        auto_halt_interval_seconds: float = 60.0,
    ) -> None:
        self._components = components
        self._config = config
        self._state: AgentState = AgentState.BOOTING
        self._transitions = Transitions()
        self.shutdown_requested: bool = False
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._auto_halt_interval_seconds = auto_halt_interval_seconds
        # S5.6 carry-fwd S7.1 reviewer MED-1: serialize halts via a
        # process-local lock so the bot, webhook, and agent-loop tick
        # cannot collide at microsecond-precision PK on simultaneous
        # halts. The lock is intra-process; cross-process serialization
        # is the journal write itself (per-row BEGIN IMMEDIATE).
        import asyncio as _asyncio
        self._halt_lock: _asyncio.Lock | None = None  # built lazily in run()
        self._auto_halt_task: _asyncio.Task | None = None
        self._discovery_task: _asyncio.Task | None = None
        self._detail_pump_task: _asyncio.Task | None = None
        # WATCHLIST — per-row throttle for the retry-blocked log (at most
        # one entry per ET day per row; process-local, so a restart may
        # re-log once — accepted, the alternative is a schema column).
        self._watchlist_retry_log_dates: dict[int, dt.date] = {}

    # -- Public surface ----------------------------------------------------

    @property
    def components(self) -> AgentComponents:
        return self._components

    @property
    def state(self) -> AgentState:
        return self._state

    async def run(self) -> int:
        """Boot, run the consumer task + reconciler, return on shutdown.

        Returns 0 on graceful shutdown; non-zero on hard-shutdown timeout
        (the only failure path that flips kill-switch to halted on the
        way out, per AC-6).
        """
        self._halt_lock = asyncio.Lock()
        install_shutdown_signal_handlers(self)

        # BOOTING — recovery scan + reconciler start.
        try:
            await self._boot()
        except Exception as e:  # noqa: BLE001
            log.error(
                "agent.boot_failed",
                extra={"event": "agent.boot_failed", "error": str(e)},
            )
            self._set_state(AgentState.SHUTTING_DOWN)
            self._set_state(AgentState.STOPPED)
            return 1

        self._set_state(AgentState.IDLE)

        consumer = asyncio.create_task(self._consume_loop(), name="agent-consumer")

        # RSS discovery + detail-fetch pump — the production-bug fix.
        # Without these tasks, `consumer` blocks forever on an empty
        # `ingestion_queue` because no producer exists. The discovery
        # loop polls EDGAR; the pump consumes `DiscoveredFiling`s,
        # fetches the primary document, persists a `FilingRow`, and
        # pushes it onto `ingestion_queue`. Both are best-effort: any
        # error inside is logged and the task continues.
        if (
            self._components.rss_discovery_loop is not None
            and self._components.detail_fetcher is not None
            and self._components.discovered_queue is not None
        ):
            await self._components.rss_discovery_loop.start()
            self._discovery_task = self._components.rss_discovery_loop._task  # type: ignore[attr-defined]
            self._detail_pump_task = asyncio.create_task(
                self._detail_pump_loop(), name="agent-detail-pump"
            )

        # S5.6 carry-fwd S7.4 + S8.2 — single 60s tick driving the
        # AutoHaltEvaluator (KS-1/2/3) and the AlertWatcher condition
        # poll. Started if either is wired; the task body checks each
        # for None internally so absent components are skipped.
        has_evaluator = (
            self._components.auto_halt_evaluator is not None
            and self._config is not None
        )
        has_watcher = self._components.alert_watcher is not None
        # WATCHLIST — the deferred-entry retry pass rides the same tick.
        has_watchlist = self._watchlist_feature_on()
        if has_evaluator or has_watcher or has_watchlist:
            self._auto_halt_task = asyncio.create_task(
                self._auto_halt_loop(), name="agent-tick"
            )

        try:
            await consumer
        except asyncio.CancelledError:
            pass

        # Reached SHUTTING_DOWN inside _consume_loop. Tear down siblings.
        rc = await self._shutdown(consumer_finished=True)
        return rc

    async def _auto_halt_loop(self) -> None:
        """S5.6 carry-fwd S7.4 MED-1 + S8.2 wiring: 60s tick that runs
        BOTH the AutoHaltEvaluator (KS-1/2/3) and the AlertWatcher's
        condition poll. Both share the tick so they stay in lockstep
        and no second timer is needed.

        - AutoHaltEvaluator: holds a single instance per the carry-fwd
          (manual-halt-then-auto-halt produces redundant rows).
          Process-local lock dedupes simultaneous halts across bot,
          webhook, and this tick.
        - AlertWatcher: snapshots latest kill_switch_state at
          construction (handled in compose_agent AFTER apply_migrations
          ran in main()) so the seed row never fires a phantom flip.
        """
        evaluator = self._components.auto_halt_evaluator
        watcher = self._components.alert_watcher
        if (
            evaluator is None
            and watcher is None
            and not self._watchlist_feature_on()
        ):
            return
        while not self.shutdown_requested:
            try:
                await asyncio.wait_for(
                    asyncio.sleep(self._auto_halt_interval_seconds),
                    timeout=self._auto_halt_interval_seconds + 1,
                )
            except (TimeoutError, asyncio.CancelledError):
                break
            if self.shutdown_requested:
                break
            now = dt.datetime.now(dt.UTC)
            if evaluator is not None and self._config is not None:
                try:
                    async with self._serialize_halt():
                        evaluator.evaluate(
                            now,
                            self._components.journal,
                            self._components.killswitch,
                            self._config.killswitch,
                        )
                except Exception as e:  # noqa: BLE001 — never kill the loop
                    log.error(
                        "agent.auto_halt_error",
                        extra={"event": "agent.auto_halt_error", "error": str(e)},
                    )
            if watcher is not None:
                try:
                    watcher.poll(now=now)
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "agent.alert_watcher_error",
                        extra={
                            "event": "agent.alert_watcher_error",
                            "error": str(e),
                        },
                    )
            # WATCHLIST — deferred-entry retry pass. Strictly AFTER the
            # halt/alert work on the same tick; internally defers to any
            # queued/in-flight day-0 pipeline work (fresh proposals keep
            # priority) and is a no-op when the feature is off or the
            # market is closed.
            try:
                await self._process_watchlist(now)
            except Exception as e:  # noqa: BLE001 — never kill the tick
                log.error(
                    "watchlist.tick_error",
                    extra={"event": "watchlist.tick_error", "error": str(e)},
                )

    async def _detail_pump_loop(self) -> None:
        """Drain `discovered_queue` → `DetailFetcher` → `ingestion_queue`.

        Bridges the RSS discovery output (`DiscoveredFiling`) into the
        consumer's input shape (`FilingRow`). Runs until shutdown is
        requested. Per-item errors are logged and the next item is
        attempted; the pump never dies on a bad filing.
        """
        from journal.repo import get_filing_by_id

        discovered = self._components.discovered_queue
        fetcher = self._components.detail_fetcher
        if discovered is None or fetcher is None:
            return
        while not self.shutdown_requested:
            try:
                discovered_filing = await discovered.get()
            except asyncio.CancelledError:
                break
            if discovered_filing is None:
                break
            if self.shutdown_requested:
                break
            try:
                filing_id = await fetcher.fetch_and_persist(discovered_filing)
            except Exception as e:  # noqa: BLE001 — never kill the pump
                log.error(
                    "agent.detail_fetch_error",
                    extra={
                        "event": "agent.detail_fetch_error",
                        "accession": discovered_filing.accession_number,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
                continue
            row = get_filing_by_id(self._components.journal.db_path, filing_id)
            if row is None:
                log.error(
                    "agent.detail_fetch_missing_row",
                    extra={
                        "event": "agent.detail_fetch_missing_row",
                        "accession": discovered_filing.accession_number,
                        "filing_id": filing_id,
                    },
                )
                continue
            # Skip partial-ingest rows: they have no usable raw_text and
            # would only burn LLM spend on an empty input. The submissions
            # reconciler (S3.4) is responsible for retrying these later.
            if row.ingest_state != "ok":
                log.info(
                    "agent.detail_fetch_partial",
                    extra={
                        "event": "agent.detail_fetch_partial",
                        "accession": discovered_filing.accession_number,
                        "filing_id": filing_id,
                        "ingest_state": row.ingest_state,
                    },
                )
                continue
            await self._components.ingestion_queue.put(row)

    def _serialize_halt(self) -> Any:
        """Process-local lock around any code path that writes a
        kill-switch row. Mitigates S7.1 reviewer MED-1 (PK collision at
        microsecond precision when bot/webhook/agent-tick collide)."""
        if self._halt_lock is None:
            self._halt_lock = asyncio.Lock()
        return self._halt_lock

    async def aclose(self) -> None:
        """Hint to release any held resources. The journal is process-
        scoped (no per-call connection cache) so this is a no-op for v1.
        """
        return None

    # -- Boot --------------------------------------------------------------

    async def _boot(self) -> None:
        log.info("agent.boot_start", extra={"event": "agent.boot_start"})
        # Reconciler may be a stub in tests; both expose .start() coroutine.
        recon_start = getattr(self._components.reconciler, "start", None)
        if recon_start is not None:
            await recon_start()
        await self._crash_recovery_scan()
        # D-087 — idempotent thesis-review re-arm sweep. A position whose
        # latest thesis-review schedule already fired without writing a
        # successor (the `adjust_stop_skipped` orphaning bug, and any
        # other gap where entry-time scheduling failed) is invisible to
        # the reconciler-driven coordinator forever. Every boot, re-arm a
        # due-now review for any OPEN BROKER position that lacks a pending
        # review. Idempotent: positions that already have a pending review
        # are skipped, so a restart loop cannot pile up schedules.
        self._rearm_orphaned_thesis_reviews()
        # PDT-TRANSITION-D-077 (ADR-012 §4.2): one-shot conversion of
        # active virtual exits to real GTC broker orders. MUST run
        # before the deferred replayer below — the converter's drain
        # invalidates `deferred_sells` rows the replayer would otherwise
        # double-sell against freshly-converted GTC orders. Runs every
        # boot regardless of `pdt_enabled` until the active set drains;
        # idempotent via `pdt-convert-{id}` client ids. Survives the P2
        # tag-removal PR (stale-prod-DB self-heal); removed post-soak.
        converter = self._components.pdt_transition_converter
        if converter is not None:
            try:
                converter.run()
            except Exception as e:  # noqa: BLE001 — defensive only.
                log.error(
                    "agent.pdt_transition_error",
                    extra={
                        "event": "agent.pdt_transition_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
        # PDT-SUNSET-2026-06-04: ADR-009 §"Pre-market deferred replay"
        # / S-PDT-5 AC-5 — one-shot drain of `deferred_sells` rows
        # whose `deferred_at` is from a prior ET trading day. Boot is
        # the seam (no separate pre-market hook in v1); the replayer
        # is idempotent on duplicate client_order_id (AC-6). Errors
        # are logged inside `run()` and never propagate.
        replayer = self._components.deferred_replayer
        if replayer is not None:
            try:
                replayer.run()
            except Exception as e:  # noqa: BLE001 — defensive only.
                log.error(
                    "agent.deferred_replayer_error",
                    extra={
                        "event": "agent.deferred_replayer_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
        log.info("agent.boot_complete", extra={"event": "agent.boot_complete"})

    async def _crash_recovery_scan(self) -> None:
        """AC-7: feed pending filings (no prefilter row) and pending
        proposals (no execution row) into the consumer queue ahead of
        any RSS arrivals. Bounded by `_RECOVERY_SCAN_MAX_FILINGS`."""
        db = self._components.journal.db_path
        with connect(db) as conn:
            pending_filings = conn.execute(
                """
                SELECT f.id FROM filings f
                LEFT JOIN prefilter_decisions p ON p.filing_id = f.id
                WHERE p.id IS NULL
                ORDER BY f.id ASC
                LIMIT ?
                """,
                (_RECOVERY_SCAN_MAX_FILINGS,),
            ).fetchall()
            pending_proposals = conn.execute(
                """
                SELECT pr.id, pr.filing_id FROM proposals pr
                LEFT JOIN executions e ON e.proposal_id = pr.id
                WHERE e.id IS NULL AND pr.kind = 'trade_proposal'
                ORDER BY pr.id ASC
                LIMIT ?
                """,
                (_RECOVERY_SCAN_MAX_FILINGS,),
            ).fetchall()

        for r in pending_filings:
            from journal.repo import get_filing_by_id

            filing = get_filing_by_id(db, int(r["id"]))
            if filing is not None:
                await self._components.ingestion_queue.put(filing)

        # For pending proposals, replay only the execution stage. We
        # tag the queue entry with a sentinel by wrapping in a tuple;
        # the consumer detects the sentinel and skips prefilter/analyze.
        for r in pending_proposals:
            from journal.repo import get_filing_by_id

            filing = get_filing_by_id(db, int(r["filing_id"]))
            if filing is None:
                continue
            await self._components.ingestion_queue.put(
                _PendingExecution(filing=filing, proposal_id=int(r["id"]))
            )

        if pending_filings or pending_proposals:
            log.info(
                "agent.recovery_scan",
                extra={
                    "event": "agent.recovery_scan",
                    "pending_filings": len(pending_filings),
                    "pending_proposals": len(pending_proposals),
                },
            )

    # -- Consumer loop -----------------------------------------------------

    async def _consume_loop(self) -> None:
        """Consume from the ingestion queue until shutdown is requested.

        AC-6: the shutdown flag is checked (a) before dequeuing a new
        filing and (b) between pipeline stages. The current in-flight
        filing always completes its pipeline before the loop exits.
        """
        queue = self._components.ingestion_queue
        while not self.shutdown_requested:
            try:
                item = await queue.get()
            except asyncio.CancelledError:
                break
            if item is None:
                # Shutdown sentinel.
                break
            if self.shutdown_requested:
                # Don't start a new pipeline once shutdown was flipped
                # while we were waiting on `queue.get()`.
                break

            cid = uuid.uuid4().hex[:12]
            try:
                from observability.log import correlation_id_scope

                ctx_mgr = correlation_id_scope(cid)
            except Exception:
                ctx_mgr = _nullcontext()  # type: ignore[assignment]

            with ctx_mgr:  # type: ignore[union-attr]
                self._set_state(AgentState.PROCESSING)
                try:
                    if isinstance(item, _PendingExecution):
                        await self._replay_execution(item.filing, item.proposal_id)
                    else:
                        await self._process_filing(item)
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "agent.pipeline_error",
                        extra={
                            "event": "agent.pipeline_error",
                            "error": str(e),
                            "error_class": type(e).__name__,
                        },
                    )
                self._set_state(AgentState.IDLE)

        self._set_state(AgentState.SHUTTING_DOWN)

    # -- Per-filing pipeline (AC-5) ---------------------------------------

    async def _process_filing(self, filing: FilingRow) -> None:
        log.info(
            "agent.pipeline_start",
            extra={
                "event": "agent.pipeline_start",
                "filing_id": filing.id,
                "accession_number": filing.accession_number,
            },
        )

        # Step 1 — prefilter.
        decision = self._components.prefilter.evaluate(filing, self._read_raw(filing))
        if decision.decision == "reject":
            log.info(
                "agent.prefilter_reject",
                extra={
                    "event": "agent.prefilter_reject",
                    "filing_id": filing.id,
                    "rule_fired": decision.rule_fired,
                },
            )
            return

        # Step 2 — analyzer + Opus (the analyzer routes high-conviction
        # to Opus internally per S5.3 + D-047).
        ctx = self._build_analyzer_context(filing)
        # Carry-forward S5.5 D-1: skip re-spend on restart.
        existing_pid = self._lookup_existing_proposal(filing, ctx)
        if existing_pid is not None:
            log.info(
                "agent.analyzer_skipped_idempotent",
                extra={
                    "event": "agent.analyzer_skipped_idempotent",
                    "filing_id": filing.id,
                    "proposal_id": existing_pid,
                },
            )
            await self._maybe_execute_existing(existing_pid)
            return

        raw_text = self._read_raw(filing)
        await self._components.analyzer.analyze(filing, raw_text, ctx)

        # Step 3 — find the proposal that was just stored. The analyzer
        # / Opus reviewer wrote it; we need the ID to drive execution.
        decision_id = self._compose_decision_id(filing)
        proposal = get_proposal_by_decision_id(
            self._components.journal.db_path, decision_id
        )
        if proposal is None or proposal.id is None:
            # Should not happen — the analyzer always persists.
            log.error(
                "agent.proposal_missing_after_analyze",
                extra={
                    "event": "agent.proposal_missing_after_analyze",
                    "decision_id": decision_id,
                },
            )
            return
        if proposal.kind != "trade_proposal":
            return  # no_trade or analyzer_malformed → no execution.

        # Step 4 — execute (validator → sizer → submitter, gated on
        # Opus review outcome).
        await self._execute(proposal.id, filing)

    async def _replay_execution(
        self, filing: FilingRow, proposal_id: int
    ) -> None:
        """AC-12: re-execute a proposal whose execution row was lost."""
        log.info(
            "agent.replay_execution",
            extra={
                "event": "agent.replay_execution",
                "filing_id": filing.id,
                "proposal_id": proposal_id,
            },
        )
        await self._execute(proposal_id, filing)

    async def _maybe_execute_existing(self, proposal_id: int) -> None:
        proposal = get_proposal_by_id(
            self._components.journal.db_path, proposal_id
        )
        if proposal is None or proposal.kind != "trade_proposal":
            return
        # Skip if executions row already exists (idempotency). A
        # `pending_capacity` row is the placeholder Feature B writes
        # when it skipped Opus; it represents "queued for retro," not
        # an executed outcome — so re-execution must NOT short-circuit
        # on it. Migration 003 allows multiple rows per proposal.
        with connect(self._components.journal.db_path) as conn:
            row = conn.execute(
                "SELECT reject_reason FROM executions "
                "WHERE proposal_id = ? "
                "AND (reject_reason IS NULL OR reject_reason != 'pending_capacity') "
                "LIMIT 1",
                (proposal_id,),
            ).fetchone()
        if row is not None:
            return
        # Re-load the source filing.
        from journal.repo import get_filing_by_id

        filing = get_filing_by_id(
            self._components.journal.db_path, proposal.filing_id
        )
        if filing is None:
            return
        await self._execute(proposal_id, filing)

    async def execute_proposal(self, proposal_id: int, filing: FilingRow) -> None:
        """Public alias so the retro-fill coordinator (Feature C) can drive
        the post-Opus chain through the same code path the consumer uses.
        """
        await self._execute(proposal_id, filing)

    async def _execute(self, proposal_id: int, filing: FilingRow) -> None:
        """Run the validator → sizer → submitter chain for `proposal_id`.

        Writes an `executions` row in every case (accepted, rejected,
        kill-switch). On reject paths, the agent loop owns the journal
        write because S6.2 / S6.3 are pure logic; on accept, S6.4's
        `OrderSubmitter.submit` writes the row.
        """
        # Idempotency: skip if already executed. A `pending_capacity`
        # row is the placeholder written when Feature B's gate skipped
        # Opus; Feature C's retro path may legitimately re-enter
        # `_execute` once Opus has reviewed, so don't treat it as a
        # terminal outcome. Migration 003 allows multiple rows per
        # proposal so a follow-up `accepted` / `rejected` insert won't
        # collide on UNIQUE.
        with connect(self._components.journal.db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM executions "
                "WHERE proposal_id = ? "
                "AND (reject_reason IS NULL OR reject_reason != 'pending_capacity') "
                "LIMIT 1",
                (proposal_id,),
            ).fetchone()
        if existing is not None:
            log.debug(
                "agent.execution_idempotent",
                extra={
                    "event": "agent.execution_idempotent",
                    "proposal_id": proposal_id,
                },
            )
            return

        proposal_row = get_proposal_by_id(
            self._components.journal.db_path, proposal_id
        )
        if proposal_row is None:
            return

        # If Opus rejected the proposal, write a rejected execution row.
        review = get_proposal_review_by_proposal_id(
            self._components.journal.db_path, proposal_id
        )
        if review is not None and review.decision in ("reject", "malformed"):
            self._write_rejected_execution(proposal_id, "opus_reject")
            return

        # Feature B / Feature C HIGH-1 fix — pending_capacity guard.
        #
        # If the proposal's conviction cleared the Opus-review threshold
        # but no `proposal_reviews` row exists, that means the analyzer's
        # capacity gate (Feature B in `SonnetAnalyzer.analyze`) skipped
        # Opus because KS-5 was at cap. We MUST NOT proceed to the
        # validator/sizer chain in that case — without Opus review the
        # trade violates FR-17, and there is a journal-vs-broker race
        # window where the sizer's KS-5 check (broker truth) could see
        # slack while Feature B (broker truth at analyze time) saw
        # capacity. Writing an `executions` row with
        # `reject_reason='pending_capacity'` (a) blocks the broker
        # submission path, (b) gives Feature C's retro-fill query a
        # deterministic signal to find these proposals, and (c) keeps
        # the audit trail complete.
        threshold = self._opus_review_threshold()
        if (
            review is None
            and threshold is not None
            and proposal_row.conviction is not None
            and proposal_row.conviction >= threshold
        ):
            log.info(
                "agent.execution_pending_capacity",
                extra={
                    "event": "agent.execution_pending_capacity",
                    "proposal_id": proposal_id,
                    "conviction": proposal_row.conviction,
                    "threshold": threshold,
                },
            )
            self._write_rejected_execution(proposal_id, "pending_capacity")
            return

        # Run the validator → sizer → submitter chain. Day-0 semantics:
        # the chain journals reject rows and (feature: watchlist) enrolls
        # capital-class rejects for a bounded deferred retry.
        await self._run_entry_chain(
            proposal_id, proposal_row, review, day_zero=True
        )

    async def _run_entry_chain(
        self,
        proposal_id: int,
        proposal_row: Any,
        review: Any,
        *,
        day_zero: bool,
    ) -> str | None:
        """Validator → sizer → submitter chain, shared by the day-0 path
        (`_execute`) and the watchlist deferred-retry path.

        `day_zero=True` preserves the exact pre-watchlist behavior:
        reject paths journal an `executions` row, the KS-7 displacement
        escape hatch may run, and capital-class rejects enroll on the
        watchlist (feature-gated). `day_zero=False` (watchlist retry)
        journals NOTHING on reject from this seam — the day-0 rejection
        row already exists and re-journaling every tick would spam the
        audit trail (the submitter still writes its own rows if a
        submission is actually attempted and fails) — and never
        displaces or re-enrolls; the caller keeps the row pending,
        bounded by expiry. The SUCCESS path is identical in both modes:
        submitter journal writes, reconciler trigger, thesis-review
        scheduling.

        Returns None when the broker accepted the submission (or a
        crashed prior submission was adopted from broker truth);
        otherwise the reject-reason string.
        """
        # Build TradeProposal pydantic model from row payload (raw_response).
        #
        # If Opus modified the proposal (decision='modify'), overlay the
        # reviewer's stop / take-profit / exit-condition changes onto the
        # payload BEFORE validation, so the validator's inverted-level +
        # exit-geometry gates, sizing, and order construction all see the
        # levels that will actually be ordered. The `proposals` row itself
        # is never touched (NFR-16 — append-only journal); the overlay
        # lives only on this in-memory working copy, and the journal
        # records the changes via `proposal_reviews.modifications_json`.
        # (`size_pct_of_capital` is intentionally NOT overlaid here — it
        # rides S5.4's working-copy `size_pct_requested` path.)
        from proposal.schemas import TradeProposal, validate_trade_proposal

        overlay: dict[str, Any] = {}
        try:
            payload = json.loads(proposal_row.raw_response)
            overlay = _reviewer_trade_overlay(review)
            if overlay:
                payload.update(overlay)
            trade = validate_trade_proposal(payload)
        except Exception as e:  # noqa: BLE001
            log.error(
                "agent.proposal_payload_invalid",
                extra={
                    "event": "agent.proposal_payload_invalid",
                    "proposal_id": proposal_id,
                    "error": str(e),
                },
            )
            if day_zero:
                self._write_rejected_execution(proposal_id, "schema")
            return "schema"
        if overlay:
            log.info(
                "execution.reviewer_overlay_applied",
                extra={
                    "event": "execution.reviewer_overlay_applied",
                    "proposal_id": proposal_id,
                    "fields": sorted(overlay),
                    "stop_loss_price": overlay.get("stop_loss_price"),
                    "take_profit_price": overlay.get("take_profit_price"),
                },
            )

        # Validator.
        validation = self._components.validator.validate(
            trade,
            self._components.broker,
            _UniverseAdapter(self._components.universe),
            self._components.killswitch,
            self._components.journal,
        )
        if isinstance(validation, Rejected):
            # Defense-in-depth WARN log when killswitch blocked us
            # (AC-5 step 5).
            if validation.reason == "kill_switch":
                log.warning(
                    "killswitch blocked proposal",
                    extra={
                        "event": "killswitch_blocked_proposal",
                        "proposal_id": proposal_id,
                        "symbol": trade.symbol,
                    },
                )
            if day_zero:
                self._write_rejected_execution(proposal_id, validation.reason)
                # WATCHLIST — `insufficient_capital` (the validator's
                # buying-power check) is the only capital-class reason
                # this branch can produce; `_maybe_enroll_watchlist`
                # filters everything else out.
                self._maybe_enroll_watchlist(
                    proposal_id, trade, validation.reason, quote_last=None
                )
            return validation.reason

        # Sizer. Hotfix 2026-05-07: also fetch pending broker orders so
        # KS-5 / KS-6 see queued pre-market entries (which haven't filled
        # yet and therefore don't appear in `get_positions()`). Without
        # this, the bot opened 28 entries in one pre-market session
        # because each new sizing check saw 0 open positions.
        account = self._components.broker.get_account()
        positions = self._components.broker.get_positions()
        try:
            open_orders = self._components.broker.get_open_orders()
        except Exception as e:  # noqa: BLE001 — never block trade flow on telemetry
            log.warning(
                "agent.get_open_orders_failed",
                extra={
                    "event": "agent.get_open_orders_failed",
                    "proposal_id": proposal_id,
                    "error": str(e),
                },
            )
            open_orders = []
        pending_buys = [o for o in open_orders if o.is_entry]
        # Hotfix 2026-07-08: KS-7 must also see the DOLLARS those pending
        # entries have committed — broker cash doesn't shrink until they
        # fill, so without this a second same-morning proposal is sized
        # against cash that is already spoken for (EOLS, −$17 cash).
        pending_entry_spend = _pending_entry_spend(
            self._components.journal.db_path, pending_buys
        )
        quote = self._components.broker.get_quote(trade.symbol)
        sizing = self._components.sizer.size(
            trade,
            account,
            positions,
            quote,
            self._config.execution if self._config else _stub_execution_config(),
            pending_buys=pending_buys,
            pending_entry_spend=pending_entry_spend,
        )
        if isinstance(sizing, SizingRejected):
            # DISPLACEMENT (default OFF) — a HIGH-conviction proposal
            # rejected by the KS-7 cash path may evict the weakest
            # qualifying open position and fund itself from the sale
            # proceeds. Every other reject reason — and every failure
            # inside the displacement path — leaves the normal rejection
            # standing, journal reason unchanged.
            displaced = None
            if day_zero and sizing.reason == "ks7_cash_reserve":
                displaced = self._attempt_displacement(
                    proposal_id=proposal_id,
                    trade=trade,
                    account=account,
                    positions=positions,
                    quote=quote,
                    pending_buys=pending_buys,
                    pending_entry_spend=pending_entry_spend,
                )
            if displaced is None:
                if day_zero:
                    self._write_rejected_execution(proposal_id, sizing.reason)
                    # WATCHLIST — ks5_concurrent_limit / ks7_cash_reserve
                    # are the sizer's capital-class reasons; the enroll
                    # helper filters the rest (ks6, conviction, one-share).
                    self._maybe_enroll_watchlist(
                        proposal_id, trade, sizing.reason, quote_last=quote.last
                    )
                return sizing.reason
            # The victim's sell is already ACCEPTED at the broker; the
            # buy below is sized against the haircut proceeds credit.
            sizing = displaced

        accepted_proposal = AcceptedProposal(
            proposal=trade,
            proposal_id=proposal_id,
            qty=sizing.qty,
            realized_dollar_size=sizing.realized_dollar_size,
            realized_pct=sizing.realized_pct,
            realized_dollar_size_request=sizing.realized_dollar_size_request,
        )

        # S5.6 carry-fwd S6.4 reviewer M-4 — orphan-order detection.
        # If a previous process crashed between broker submission and
        # the journal write, the broker still has the order under its
        # deterministic `prop-{id}-{role}` client_order_id. Re-submit
        # would create a duplicate. Reconstruct the journal rows from
        # broker truth and short-circuit re-submission.
        if self._adopt_orphan_orders(accepted_proposal):
            return None

        try:
            result = self._components.submitter.submit(
                accepted_proposal,
                self._components.broker,
                self._components.journal,
                self._components.killswitch,
                # PDT-SUNSET-2026-06-04: ADR-009 §3.1 — branch on the
                # process-shared activation flag. None on legacy/test
                # paths short-circuits to the existing pre-placement
                # behavior.
                pdt_state=self._components.pdt_state,
            )
        except (BrokerRejected, BrokerUnavailable) as e:
            # The submitter writes journal rows for both branches and
            # halts the killswitch on stop-leg failures. We log here as
            # defense-in-depth so the audit trail still records anything
            # that escaped the submitter's handler.
            log.error(
                "agent.broker_unavailable",
                extra={
                    "event": "agent.broker_unavailable",
                    "proposal_id": proposal_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # The submitter writes a `submission_failed` row on broker
            # outage; if the exception escaped that path entirely, still
            # record so the auditor sees it. A `pending_capacity`
            # placeholder doesn't count as "the submitter wrote a row"
            # — we still need a broker_unavailable row in that case.
            # (Day-0 only: a watchlist retry always has its day-0 reject
            # row already, and the retry stays pending regardless.)
            if day_zero:
                with connect(self._components.journal.db_path) as conn:
                    row = conn.execute(
                        "SELECT 1 FROM executions "
                        "WHERE proposal_id = ? "
                        "AND (reject_reason IS NULL OR reject_reason != 'pending_capacity') "
                        "LIMIT 1",
                        (proposal_id,),
                    ).fetchone()
                if row is None:
                    self._write_rejected_execution(proposal_id, "broker_unavailable")
            return "broker_unavailable"

        # S5.6 carry-fwd S6.5 reviewer M-2 (HIGH): trigger the position
        # reconciler after a SUCCESSFUL submission so the operator's
        # view of state catches up immediately (FR-24 behavior side).
        # Only fire on Accepted; partial-no-stop and submission_failed
        # already self-flagged via the kill-switch / journal path.
        if isinstance(result, SubmissionAccepted):
            trigger = getattr(
                self._components.reconciler,
                "trigger_after_submission",
                None,
            )
            if trigger is not None:
                try:
                    trigger()
                except Exception as e:  # noqa: BLE001 — never block trade flow
                    log.warning(
                        "agent.reconciler_trigger_failed",
                        extra={
                            "event": "agent.reconciler_trigger_failed",
                            "proposal_id": proposal_id,
                            "error": str(e),
                        },
                    )

            # Feature A — schedule the time-horizon thesis review. The
            # proposal's `time_horizon_days` field drives the due date
            # (default 14 days if missing). A schedule row written here
            # arms the reconciler-driven coordinator; if entry fails or
            # stop/TP fires before the date, the coordinator's open-
            # position guard discards the schedule silently.
            try:
                self._schedule_thesis_review(
                    execution_id=result.execution_id,
                    proposal=trade,
                )
            except Exception as e:  # noqa: BLE001 — must not block trade flow
                log.warning(
                    "agent.thesis_schedule_failed",
                    extra={
                        "event": "agent.thesis_schedule_failed",
                        "proposal_id": proposal_id,
                        "execution_id": result.execution_id,
                        "error": str(e),
                    },
                )
            return None

        # SubmissionFailed — the submitter journaled its own row (and
        # halted the kill-switch on stop-leg failures); just surface the
        # outcome to callers (watchlist retry keeps the row pending).
        return "submission_failed"

    # -- Watchlist (deferred entry) ----------------------------------------

    def _watchlist_feature_on(self) -> bool:
        from app.watchlist import watchlist_enabled

        cfg = getattr(self._config, "execution", None)
        return cfg is not None and watchlist_enabled(cfg)

    def _maybe_enroll_watchlist(
        self,
        proposal_id: int,
        trade: Any,
        reason: str,
        *,
        quote_last: float | None,
    ) -> None:
        """WATCHLIST enrollment — park an approved, capital-blocked,
        high-conviction proposal for a bounded deferred retry.

        Called ONLY from the day-0 reject paths of `_run_entry_chain`,
        which sits after the Opus-reject and pending_capacity guards —
        so every candidate here is an approved proposal. The reason
        filter narrows to the CAPITAL class only
        (`app.watchlist.CAPITAL_REJECT_REASONS`): validator rejects
        other than `insufficient_capital`, Opus rejects and kill-switch
        halts never reach an insert. Best-effort: any error degrades to
        "not enrolled" and never disturbs the journaled rejection.

        `quote_last` is the decision-time NBBO last the sizer used
        (loop fetches it right before sizing); None on the
        validator-stage `insufficient_capital` path, where the same
        quote source is fetched here instead — either way
        `reference_price` documents the best available decision-time
        price for the chase guard.
        """
        try:
            from app.watchlist import (
                CAPITAL_REJECT_REASONS,
                compute_expiry,
                watchlist_enabled,
            )

            cfg = getattr(self._config, "execution", None)
            if cfg is None or not watchlist_enabled(cfg):
                return
            if reason not in CAPITAL_REJECT_REASONS:
                return
            if trade.conviction < cfg.watchlist_min_conviction:
                return
            # Symbol-not-held guard, using the same effective-symbol view
            # the sizer's KS-5/KS-6 checks use (broker positions plus
            # queued pre-market entry buys).
            held = {
                p.symbol
                for p in self._components.broker.get_positions()
                if p.qty != 0
            }
            try:
                held |= {
                    o.symbol
                    for o in self._components.broker.get_open_orders()
                    if o.is_entry
                }
            except Exception:  # noqa: BLE001 — positions are the hard gate
                pass
            if trade.symbol in held:
                return
            from journal.models import WatchlistRow
            from journal.repo import (
                WatchlistDuplicatePending,
                has_pending_watchlist_for_symbol,
                insert_watchlist_row,
            )

            db = self._components.journal.db_path
            if has_pending_watchlist_for_symbol(db, trade.symbol):
                return  # one active row per symbol
            if quote_last is None:
                quote_last = self._components.broker.get_quote(trade.symbol).last
            now = dt.datetime.now(dt.UTC)
            expires_at = compute_expiry(now, cfg.watchlist_expiry_trading_days)
            try:
                watchlist_id = insert_watchlist_row(
                    db,
                    WatchlistRow(
                        proposal_id=proposal_id,
                        symbol=trade.symbol,
                        conviction=trade.conviction,
                        reference_price=quote_last,
                        expires_at=expires_at,
                        notes=f"reject_reason={reason}",
                    ),
                )
            except WatchlistDuplicatePending:
                return  # lost a race with a concurrent enrollment — benign
            log.info(
                "watchlist.added",
                extra={
                    "event": "watchlist.added",
                    "watchlist_id": watchlist_id,
                    "proposal_id": proposal_id,
                    "symbol": trade.symbol,
                    "conviction": trade.conviction,
                    "reference_price": quote_last,
                    "expires_at": expires_at.isoformat(),
                    "reject_reason": reason,
                },
            )
        except Exception as e:  # noqa: BLE001 — the rejection must stand
            log.error(
                "watchlist.enroll_error",
                extra={
                    "event": "watchlist.enroll_error",
                    "proposal_id": proposal_id,
                    "error": str(e),
                },
            )

    async def _process_watchlist(self, now: dt.datetime) -> None:
        """WATCHLIST — one deferred-entry retry pass (oldest row first).

        Ordering / safety contract:
          - feature off (`watchlist_min_conviction = 0`) → return before
            any journal read: no table writes, no tick work;
          - regular market hours only (entries are day orders — retrying
            off-hours would queue stale-priced entries);
          - PRIORITY: if the consumer has queued or in-flight day-0
            pipeline work, the whole pass defers to the next tick so
            fresh same-day signals are always processed first;
          - per row: expiry (trading days, stamped at enrollment) →
            held-symbol skip → chase guard (strictly above the ceiling
            skips; exactly AT the ceiling is allowed) → entry through
            the NORMAL execution chain. A failed entry leaves the row
            pending — no status change, and the retry-blocked log fires
            at most once per ET day per row.
        """
        from app.watchlist import chase_exceeded, ensure_utc, watchlist_enabled
        from config.calendar import ET, is_market_hours
        from execution.displacement import trading_days_held

        cfg = getattr(self._config, "execution", None)
        if cfg is None or not watchlist_enabled(cfg):
            return
        if not is_market_hours(now):
            return
        if (
            not self._components.ingestion_queue.empty()
            or self._state == AgentState.PROCESSING
        ):
            return

        from journal.repo import get_pending_watchlist, resolve_watchlist_row

        db = self._components.journal.db_path
        rows = get_pending_watchlist(db)
        if not rows:
            return
        try:
            held = {
                p.symbol
                for p in self._components.broker.get_positions()
                if p.qty != 0
            }
        except Exception as e:  # noqa: BLE001 — broker glitch: retry next tick
            log.warning(
                "watchlist.positions_unavailable",
                extra={
                    "event": "watchlist.positions_unavailable",
                    "error": str(e),
                },
            )
            return

        for row in rows:
            if row.id is None:  # pragma: no cover — SELECT always has ids
                continue
            if now >= ensure_utc(row.expires_at):
                resolve_watchlist_row(db, row.id, status="expired")
                self._watchlist_retry_log_dates.pop(row.id, None)
                log.info(
                    "watchlist.expired",
                    extra={
                        "event": "watchlist.expired",
                        "watchlist_id": row.id,
                        "proposal_id": row.proposal_id,
                        "symbol": row.symbol,
                    },
                )
                continue
            if row.symbol in held:
                resolve_watchlist_row(db, row.id, status="skipped_held")
                self._watchlist_retry_log_dates.pop(row.id, None)
                log.info(
                    "watchlist.skipped_held",
                    extra={
                        "event": "watchlist.skipped_held",
                        "watchlist_id": row.id,
                        "proposal_id": row.proposal_id,
                        "symbol": row.symbol,
                    },
                )
                continue
            try:
                quote = self._components.broker.get_quote(row.symbol)
            except Exception as e:  # noqa: BLE001 — leave pending, next tick
                log.warning(
                    "watchlist.quote_unavailable",
                    extra={
                        "event": "watchlist.quote_unavailable",
                        "watchlist_id": row.id,
                        "symbol": row.symbol,
                        "error": str(e),
                    },
                )
                continue
            ceiling = row.reference_price * (
                1.0 + cfg.watchlist_max_chase_pct / 100.0
            )
            if chase_exceeded(
                quote.last, row.reference_price, cfg.watchlist_max_chase_pct
            ):
                resolve_watchlist_row(
                    db,
                    row.id,
                    status="skipped_chase",
                    notes=(
                        f"last={quote.last:.4f} ref={row.reference_price:.4f} "
                        f"ceiling={ceiling:.4f}"
                    ),
                )
                self._watchlist_retry_log_dates.pop(row.id, None)
                log.info(
                    "watchlist.skipped_chase",
                    extra={
                        "event": "watchlist.skipped_chase",
                        "watchlist_id": row.id,
                        "proposal_id": row.proposal_id,
                        "symbol": row.symbol,
                        "reference_price": row.reference_price,
                        "current_last": quote.last,
                        "chase_ceiling": ceiling,
                        "max_chase_pct": cfg.watchlist_max_chase_pct,
                    },
                )
                continue

            reason = await self._attempt_watchlist_entry(row)
            if reason is None:
                created = ensure_utc(row.created_at) if row.created_at else now
                lag_days = trading_days_held(
                    created.astimezone(ET).date(), now.astimezone(ET).date()
                )
                delta_pct = (
                    (quote.last - row.reference_price)
                    / row.reference_price
                    * 100.0
                )
                resolve_watchlist_row(
                    db,
                    row.id,
                    status="entered",
                    notes=(
                        f"lag_trading_days={lag_days} "
                        f"entry_last={quote.last:.4f} "
                        f"delta_pct={delta_pct:+.2f}"
                    ),
                )
                self._watchlist_retry_log_dates.pop(row.id, None)
                held.add(row.symbol)
                log.info(
                    "watchlist.entered",
                    extra={
                        "event": "watchlist.entered",
                        "watchlist_id": row.id,
                        "proposal_id": row.proposal_id,
                        "symbol": row.symbol,
                        "lag_trading_days": lag_days,
                        "reference_price": row.reference_price,
                        "entry_last": quote.last,
                        "price_delta_pct": delta_pct,
                    },
                )
            else:
                # Still blocked (usually capital again) — leave pending
                # with NO status change; throttle the log to once per ET
                # day per row so a full book doesn't spam every tick.
                today_et = now.astimezone(ET).date()
                if self._watchlist_retry_log_dates.get(row.id) != today_et:
                    self._watchlist_retry_log_dates[row.id] = today_et
                    log.info(
                        "watchlist.retry_blocked",
                        extra={
                            "event": "watchlist.retry_blocked",
                            "watchlist_id": row.id,
                            "proposal_id": row.proposal_id,
                            "symbol": row.symbol,
                            "reason": reason,
                        },
                    )

    async def _attempt_watchlist_entry(self, row: Any) -> str | None:
        """Retry a parked proposal through the NORMAL execution chain.

        DETERMINISTIC — no LLM calls anywhere on this path: the Opus
        review already happened on day 0 (enrollment only ever follows
        an approved proposal), and the chain is validator + sizer +
        submitter only. The validator IS re-run — it is pure, cheap
        (one quote + one account snapshot, zero LLM) and deterministic —
        so kill-switch, universe, price-floor and exit-geometry are all
        re-checked at retry-time prices; the sizer re-runs every
        KS-4/5/6/7 cap, including symbol-not-held (KS-6, on the
        positions ∪ pending-entries view).

        Returns None on an accepted submission; a reason string on any
        failure — the caller leaves the row pending (bounded by expiry).
        """
        db = self._components.journal.db_path
        proposal_row = get_proposal_by_id(db, row.proposal_id)
        if proposal_row is None:
            return "proposal_missing"
        # Never double-enter — GATING RULE (review A1):
        #   - any non-rejected, non-submission_failed execution row
        #     (accepted / submission_partial_no_stop / adopted-from-
        #     broker) means the entry already exists → gated forever;
        #   - a `submission_failed` row gates ONLY when at least one
        #     `orders` row hangs off it — i.e. some leg actually reached
        #     the broker (the 4cdf3d9 duplicate-client_order_id incident
        #     class) and a blind re-submit could double-enter;
        #   - a `submission_failed` row with NO orders rows means nothing
        #     reached the broker (transient BrokerUnavailable, atomic
        #     bracket rejection — the submitter journals these with
        #     submitted_orders_json='[]' and writes no order rows), so
        #     the row stays retryable instead of stranding pending until
        #     silent expiry.
        # Belt-and-braces for the no-orders-row path: the chain's
        # `_adopt_orphan_orders` still probes the broker for a live
        # `prop-{id}-entry` before any re-submission.
        with connect(db) as conn:
            existing = conn.execute(
                "SELECT 1 FROM executions e "
                "WHERE e.proposal_id = ? "
                "AND e.decision != 'rejected' "
                "AND (e.decision != 'submission_failed' "
                "     OR EXISTS (SELECT 1 FROM orders o "
                "                WHERE o.execution_id = e.id)) "
                "LIMIT 1",
                (row.proposal_id,),
            ).fetchone()
        if existing is not None:
            return "already_executed"
        review = get_proposal_review_by_proposal_id(db, row.proposal_id)
        if review is not None and review.decision in ("reject", "malformed"):
            return "opus_reject"  # defensive; enrollment never follows these
        return await self._run_entry_chain(
            row.proposal_id, proposal_row, review, day_zero=False
        )

    # -- Helpers -----------------------------------------------------------

    def _opus_review_threshold(self) -> int | None:
        """Return the conviction threshold above which Opus review is
        mandatory before broker submission. Sourced from
        `config.analyzer.opus_review_conviction_threshold`. Returns None
        when no config is wired (test convenience) — the
        `pending_capacity` guard then degrades to a no-op so legacy
        unit tests that exercise `_execute` in isolation still pass.
        """
        if self._config is None:
            return None
        try:
            return int(self._config.analyzer.opus_review_conviction_threshold)
        except (AttributeError, TypeError, ValueError):
            return None

    def _attempt_displacement(
        self,
        *,
        proposal_id: int,
        trade: Any,
        account: Any,
        positions: list[Any],
        quote: Any,
        pending_buys: list[OpenOrder],
        pending_entry_spend: float,
    ) -> SizingAccepted | None:
        """DISPLACEMENT — deterministic KS-7 escape hatch (default OFF).

        When a high-conviction proposal cannot fund one share above the
        KS-7 reserve floor, evict the weakest qualifying open position
        and size the entry against the haircut sale proceeds (see
        `execution.displacement`). Returns the displacement
        `SizingAccepted` — the victim's sell already ACCEPTED at the
        broker — or None, in which case the caller journals the normal
        `ks7_cash_reserve` rejection, reason unchanged. Never raises:
        any error inside the path degrades to the normal rejection.
        """
        cfg = getattr(self._config, "execution", None)
        if cfg is None or not getattr(cfg, "displacement_enabled", False):
            return None
        try:
            from execution.displacement import DisplacementEngine

            engine = DisplacementEngine(
                db_path=self._components.journal.db_path,
                broker=self._components.broker,
            )
            return engine.attempt(
                proposal=trade,
                proposal_id=proposal_id,
                account=account,
                open_positions=positions,
                quote=quote,
                cfg=cfg,
                sizer=self._components.sizer,
                pending_buys=pending_buys,
                pending_entry_spend=pending_entry_spend,
            )
        except Exception as e:  # noqa: BLE001 — the rejection must stand
            log.error(
                "execution.displacement.error",
                extra={
                    "event": "execution.displacement.error",
                    "proposal_id": proposal_id,
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
            return None

    def _rearm_orphaned_thesis_reviews(self) -> None:
        """D-087 — boot-time, idempotent re-arm of orphaned thesis
        reviews.

        For every OPEN broker position whose `accepted` execution has no
        *pending* thesis review (latest schedule already fired, or none
        ever written), insert a due-now `thesis_review_schedule` row so
        the reconciler-driven coordinator picks it back up on its next
        tick. Broker positions are the source of truth for "open", so a
        closed slot can never be armed even if a stale journal `positions`
        row lingers.

        Idempotent: executions that already have a pending review are not
        returned by the repo query, so repeated boots never stack
        duplicate schedules. Best-effort: any failure is logged and boot
        proceeds (a missed re-arm is recoverable next boot; a crashed
        boot is not).
        """
        from journal.models import ThesisReviewScheduleRow
        from journal.repo import (
            find_accepted_executions_without_pending_thesis_review,
            insert_thesis_review_schedule,
        )

        db = self._components.journal.db_path
        try:
            candidates = find_accepted_executions_without_pending_thesis_review(db)
        except Exception as e:  # noqa: BLE001 — never block boot on the sweep
            log.error(
                "agent.thesis_rearm_query_failed",
                extra={"event": "agent.thesis_rearm_query_failed", "error": str(e)},
            )
            return
        if not candidates:
            return

        try:
            open_symbols = {
                p.symbol for p in self._components.broker.get_positions() if p.qty != 0
            }
        except Exception as e:  # noqa: BLE001 — broker glitch: skip, retry next boot
            log.warning(
                "agent.thesis_rearm_positions_failed",
                extra={
                    "event": "agent.thesis_rearm_positions_failed",
                    "error": str(e),
                },
            )
            return

        now = dt.datetime.now(dt.UTC)
        rearmed = 0
        for execution_id, symbol in candidates:
            if symbol not in open_symbols:
                continue
            try:
                insert_thesis_review_schedule(
                    db,
                    ThesisReviewScheduleRow(
                        execution_id=execution_id,
                        due_at=now,
                        scheduled_reason="rearm",
                    ),
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "agent.thesis_rearm_insert_failed",
                    extra={
                        "event": "agent.thesis_rearm_insert_failed",
                        "execution_id": execution_id,
                        "symbol": symbol,
                        "error": str(e),
                    },
                )
                continue
            rearmed += 1
            log.info(
                "agent.thesis_review_rearmed",
                extra={
                    "event": "agent.thesis_review_rearmed",
                    "execution_id": execution_id,
                    "symbol": symbol,
                },
            )
        if rearmed:
            log.info(
                "agent.thesis_rearm_sweep",
                extra={
                    "event": "agent.thesis_rearm_sweep",
                    "rearmed": rearmed,
                    "candidates": len(candidates),
                },
            )

    def _schedule_thesis_review(
        self,
        *,
        execution_id: int,
        proposal: Any,
    ) -> None:
        """Feature A — write the entry-time thesis-review schedule row.

        Due date is `now + proposal.time_horizon_days` days. Calling code
        guards on success of the order submission, so failures here are
        non-fatal — log and proceed.
        """
        from journal.models import ThesisReviewScheduleRow
        from journal.repo import insert_thesis_review_schedule

        horizon = getattr(proposal, "time_horizon_days", None)
        if not isinstance(horizon, int) or horizon <= 0:
            horizon = 14  # Feature A default per task spec
        # Cap the FIRST review so a fresh position is re-evaluated at least
        # as often as the configured ceiling (default weekly), even when the
        # analyzer declared a long catalyst horizon. Subsequent reviews
        # already recur weekly via HOLD_RESCHEDULE_DAYS, so this only bounds
        # the entry-time interval. When no config is wired (legacy/test
        # construction) the declared horizon is used uncapped.
        review_days = horizon
        cap = self._max_initial_review_days()
        if cap is not None:
            review_days = min(horizon, cap)
        due = dt.datetime.now(dt.UTC) + dt.timedelta(days=review_days)
        insert_thesis_review_schedule(
            self._components.journal.db_path,
            ThesisReviewScheduleRow(
                execution_id=execution_id,
                due_at=due,
                scheduled_reason="entry",
            ),
        )
        log.info(
            "agent.thesis_review_scheduled",
            extra={
                "event": "agent.thesis_review_scheduled",
                "execution_id": execution_id,
                "horizon_days": horizon,
                "review_days": review_days,
                "due_at": due.isoformat(),
            },
        )

    def _max_initial_review_days(self) -> int | None:
        """First-review cap (Feature A): the configured ceiling on the
        entry-time thesis-review interval, or `None` when no config is wired.

        Returns `None` for legacy/test construction (`config=None` or a stub
        lacking the field) so the declared horizon is used uncapped, which
        preserves prior behavior. In production `compose_agent` always passes
        the `AppConfig`, so the `ExecutionConfig.max_initial_review_days`
        default (7) applies unless the operator overrides it.
        """
        cfg = self._config
        if cfg is None:
            return None
        execution = getattr(cfg, "execution", None)
        cap = getattr(execution, "max_initial_review_days", None)
        return cap if isinstance(cap, int) and cap > 0 else None

    def _read_raw(self, filing: FilingRow) -> str:
        try:
            with open(filing.raw_text_path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            log.warning(
                "agent.raw_text_missing",
                extra={
                    "event": "agent.raw_text_missing",
                    "filing_id": filing.id,
                    "path": filing.raw_text_path,
                },
            )
            return ""

    def _build_analyzer_context(self, filing: FilingRow) -> AnalyzerContext:
        try:
            ks_state = self._components.killswitch.state().state
        except KillSwitchUninitialized:
            ks_state = "halted"  # fail closed
        # S5.6 carry-fwd S5.1 A-1: AnalyzerContext.kill_switch_state is a
        # Literal["halted","ok"]; collapse the v1 schema's vocabulary
        # ("active" → "ok", anything halted → "halted").
        ks_literal: Any = "halted" if ks_state == "halted" else "ok"
        # `open_positions_count` here is JOURNAL-DERIVED (latest
        # `positions` snapshot per symbol with qty != 0). It is used for
        # PROMPT CONTEXT only — the LLM sees a number to inform its
        # reasoning about portfolio state. Feature B's KS-5 capacity
        # gate consults a SEPARATE BROKER-DERIVED counter
        # (`open_positions_counter` wired by `compose_agent` to
        # `lambda: len(broker.get_positions())`) so it agrees with
        # `SizingEngine.size()` at sizing.py:103. The two values can
        # diverge briefly (reconciler hasn't ticked since a stop fired)
        # but the gate uses broker truth, not this count.
        positions = self._components.journal.get_open_positions()
        # Universe summary is an issuer-membership snapshot string for
        # block-2 (cached, daily-stable). Keep it minimal at v1.
        universe_summary = self._summarize_universe(filing)
        return AnalyzerContext(
            universe_summary=universe_summary,
            kill_switch_state=ks_literal,
            open_positions_count=len(positions),
            decision_id=self._compose_decision_id(filing),
        )

    def _summarize_universe(self, filing: FilingRow) -> str:
        member = (
            self._components.universe.get_member(filing.issuer_ticker)
            if filing.issuer_ticker
            else None
        )
        if member is None:
            return f"issuer={filing.issuer_ticker} (not in current universe member list)"
        return (
            f"issuer={member.ticker} cik={member.cik} exchange={member.exchange} "
            f"market_cap={member.market_cap:.0f} prev_close={member.prev_close:.2f}"
        )

    def _compose_decision_id(self, filing: FilingRow) -> str:
        """Mirror SonnetAnalyzer's decision_id derivation so the loop
        can look up the persisted proposal after analyze() returns.
        The prompt name MUST be the constant the builder resolves for
        the analyzer (ACTIVE_SONNET_ANALYSIS_PROMPT) — a diverging
        literal here silently mismatches decision ids."""
        from analyzer.sonnet import compute_decision_id
        from prompts.loader import ACTIVE_SONNET_ANALYSIS_PROMPT

        sonnet_model_id = self._components.analyzer._sonnet_model_id  # noqa: SLF001
        prompt_version = self._components.analyzer._builder.prompt_version(  # noqa: SLF001
            ACTIVE_SONNET_ANALYSIS_PROMPT
        )
        return compute_decision_id(
            filing_id=filing.id,
            model_id=sonnet_model_id,
            prompt_version=prompt_version,
        )

    def _lookup_existing_proposal(
        self, filing: FilingRow, ctx: AnalyzerContext
    ) -> int | None:
        decision_id = ctx.decision_id
        existing = get_proposal_by_decision_id(
            self._components.journal.db_path, decision_id
        )
        if existing is None or existing.id is None:
            return None
        return existing.id

    def _adopt_orphan_orders(self, ap: AcceptedProposal) -> bool:
        """S5.6 carry-fwd S6.4 reviewer M-4 — adopt broker-side orphans.

        Returns True if the broker already has at least the entry order
        for this proposal (deterministic `prop-{id}-entry` client_order
        id). In that case we DO NOT re-submit; instead we materialize
        the journal rows from broker truth so the agent's view of state
        catches up. Returns False (no orphan) → caller proceeds with
        normal submission.

        Conservative: any error during the lookup falls back to "no
        orphan" so we don't block a legitimate submission on a broker
        glitch. The 5-second-window crash this guards against is rare
        enough that a missed adoption is acceptable; a duplicate order
        is not.
        """
        try:
            entry = self._components.broker.get_order_by_client_id(
                f"prop-{ap.proposal_id}-entry"
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "agent.orphan_lookup_failed",
                extra={
                    "event": "agent.orphan_lookup_failed",
                    "proposal_id": ap.proposal_id,
                    "error": str(e),
                },
            )
            return False
        if entry is None:
            return False
        # Review A2 — liveness check. The broker remembers TERMINAL-DEAD
        # orders under their client ids too: a crash-window entry that was
        # rejected / canceled / expired WITHOUT any fill must NOT be
        # adopted as a phantom position (it created no exposure and never
        # will); it falls through to normal submission. (If the broker
        # refuses the reused client_order_id, the submitter journals a
        # `submission_failed` and — on the watchlist path — the row stays
        # pending, bounded by expiry.)
        #
        # A2 follow-up (real-money edge): "canceled" / "expired" can carry
        # a PARTIAL fill — those shares are REAL exposure sitting without
        # protective legs, so a dead order WITH fills must still be
        # adopted (the partial_no_stop halt path below then flags it),
        # never skipped-and-re-submitted, which would double the position.
        # Only status-dead AND zero-filled means "safe to treat as absent".
        if (
            entry.status in _TERMINAL_DEAD_ORDER_STATUSES
            and (entry.filled_qty or 0) == 0
        ):
            log.info(
                "agent.orphan_entry_dead",
                extra={
                    "event": "agent.orphan_entry_dead",
                    "proposal_id": ap.proposal_id,
                    "symbol": entry.symbol,
                    "status": entry.status,
                },
            )
            return False

        # Broker has the entry. Pull stop + tp too — both legs may have
        # been submitted before the crash.
        try:
            stop = self._components.broker.get_order_by_client_id(
                f"prop-{ap.proposal_id}-stop"
            )
            tp = self._components.broker.get_order_by_client_id(
                f"prop-{ap.proposal_id}-tp"
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "agent.orphan_followup_lookup_failed",
                extra={
                    "event": "agent.orphan_followup_lookup_failed",
                    "proposal_id": ap.proposal_id,
                    "error": str(e),
                },
            )
            stop = None
            tp = None

        # Materialize execution + order rows from broker truth.
        submitted_payload: list[dict[str, Any]] = [
            {"broker_order_id": entry.broker_order_id, "role": "entry"}
        ]
        if stop is not None:
            submitted_payload.append(
                {"broker_order_id": stop.broker_order_id, "role": "stop"}
            )
        if tp is not None:
            submitted_payload.append(
                {"broker_order_id": tp.broker_order_id, "role": "take_profit"}
            )

        # Decision = "accepted" if both entry and stop are at the
        # broker; "submission_partial_no_stop" if only entry made it
        # (consistent with S6.4's failure-handling vocabulary).
        decision = (
            "accepted" if stop is not None else "submission_partial_no_stop"
        )
        execution_id = self._components.journal.insert_execution(
            ExecutionRow(
                proposal_id=ap.proposal_id,
                decision=decision,
                realized_size_pct=ap.realized_pct,
                realized_dollar_size=ap.realized_dollar_size,
                submitted_orders_json=json.dumps(submitted_payload),
            )
        )
        # Journal each leg the broker remembers.
        from journal.models import OrderRow

        self._components.journal.insert_order(
            OrderRow(
                execution_id=execution_id,
                role="entry",
                symbol=entry.symbol,
                side=entry.side,
                order_type=entry.order_type,
                qty=entry.qty,
                limit_price=entry.limit_price,
                stop_price=entry.stop_price,
                tif="day",  # entries are always day per S6.4
                broker_order_id=entry.broker_order_id,
                submitted_at=entry.submitted_at,
                final_status=entry.status,
                notes="adopted_from_broker_on_recovery",
            )
        )
        if stop is not None:
            self._components.journal.insert_order(
                OrderRow(
                    execution_id=execution_id,
                    role="stop",
                    symbol=stop.symbol,
                    side=stop.side,
                    order_type=stop.order_type,
                    qty=stop.qty,
                    limit_price=stop.limit_price,
                    stop_price=stop.stop_price,
                    tif="gtc",
                    broker_order_id=stop.broker_order_id,
                    submitted_at=stop.submitted_at,
                    final_status=stop.status,
                    notes="adopted_from_broker_on_recovery",
                )
            )
        if tp is not None:
            self._components.journal.insert_order(
                OrderRow(
                    execution_id=execution_id,
                    role="take_profit",
                    symbol=tp.symbol,
                    side=tp.side,
                    order_type=tp.order_type,
                    qty=tp.qty,
                    limit_price=tp.limit_price,
                    stop_price=tp.stop_price,
                    tif="gtc",
                    broker_order_id=tp.broker_order_id,
                    submitted_at=tp.submitted_at,
                    final_status=tp.status,
                    notes="adopted_from_broker_on_recovery",
                )
            )

        # Critical state guard: if entry exists but stop does not, halt
        # the kill-switch (mirrors S6.4's submission_partial_no_stop
        # handling).
        if stop is None:
            try:
                self._components.killswitch.halt(
                    reason="submission_partial_no_stop",
                    set_by="system",
                    notes=(
                        f"adopted entry {entry.broker_order_id} for "
                        f"{entry.symbol} on recovery; broker had no stop"
                    ),
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "agent.orphan_halt_failed",
                    extra={
                        "event": "agent.orphan_halt_failed",
                        "proposal_id": ap.proposal_id,
                        "error": str(e),
                    },
                )

        log.warning(
            "agent.adopted_orphan_orders",
            extra={
                "event": "agent.adopted_orphan_orders",
                "proposal_id": ap.proposal_id,
                "execution_id": execution_id,
                "decision": decision,
                "legs_adopted": len(submitted_payload),
            },
        )
        return True

    def _write_rejected_execution(self, proposal_id: int, reason: str) -> None:
        """Persist the journal-side reject row for FR-22.

        S5.6 carry-fwd S6.4 reviewer M-1 (HIGH): S6.4's `OrderSubmitter`
        only handles ACCEPTED proposals; the validator/sizer reject
        paths never reach a journal seam from inside execution. The
        agent loop owns this write so `executions.reject_reason` is the
        canonical, queryable reason for daily-report cost-overrun and
        kill-switch-blocked counts. `submitted_orders_json="[]"` is
        explicit so downstream queries don't need to special-case NULL
        for the rejected branch.
        """
        self._components.journal.insert_execution(
            ExecutionRow(
                proposal_id=proposal_id,
                decision="rejected",
                reject_reason=reason,
                submitted_orders_json="[]",
            )
        )
        log.info(
            "agent.execution_rejected",
            extra={
                "event": "agent.execution_rejected",
                "proposal_id": proposal_id,
                "reject_reason": reason,
            },
        )

    # -- State machine + shutdown -----------------------------------------

    def _set_state(self, new: AgentState) -> None:
        """Drive a state transition. AC-3: illegal moves raise."""
        if new == self._state:
            return
        self._transitions.assert_legal(self._state, new)
        prev = self._state
        self._state = new
        log.info(
            "agent.state_transition",
            extra={
                "event": "agent.state_transition",
                "from_state": prev.value,
                "to_state": new.value,
            },
        )

    async def _shutdown(self, *, consumer_finished: bool) -> int:
        """Graceful teardown after the consumer has reached SHUTTING_DOWN.

        Returns 0 on clean shutdown, 1 on hard-shutdown timeout (writes
        a defensive halt row before exit per AC-6).
        """
        deadline = time.monotonic() + self._shutdown_grace_seconds

        # Cancel the auto-halt tick task.
        if self._auto_halt_task is not None and not self._auto_halt_task.done():
            self._auto_halt_task.cancel()
            try:
                await self._auto_halt_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Stop RSS discovery (cancels its task and persists the cursor).
        rss = self._components.rss_discovery_loop
        if rss is not None:
            try:
                await asyncio.wait_for(
                    rss.stop(),
                    timeout=max(1.0, deadline - time.monotonic()),
                )
            except TimeoutError:
                log.error(
                    "agent.rss_stop_timeout",
                    extra={"event": "agent.rss_stop_timeout"},
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "agent.rss_stop_error",
                    extra={"event": "agent.rss_stop_error", "error": str(e)},
                )

        # Cancel the detail-pump task (it's blocked on `discovered_queue.get()`).
        if (
            self._detail_pump_task is not None
            and not self._detail_pump_task.done()
        ):
            self._detail_pump_task.cancel()
            try:
                await self._detail_pump_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Stop reconciler.
        recon_stop = getattr(self._components.reconciler, "stop", None)
        if recon_stop is not None:
            try:
                await asyncio.wait_for(
                    recon_stop(), timeout=max(0.0, deadline - time.monotonic())
                )
            except (TimeoutError, asyncio.TimeoutError):
                log.error(
                    "agent.reconciler_stop_timeout",
                    extra={"event": "agent.reconciler_stop_timeout"},
                )
                self._hard_shutdown_halt()
                self._set_state(AgentState.STOPPED)
                return 1

        self._set_state(AgentState.STOPPED)
        return 0

    def _hard_shutdown_halt(self) -> None:
        """Defensive halt-state insert per AC-6 final clause."""
        from journal.models import KillSwitchStateRow

        try:
            self._components.journal.insert_kill_switch_state(
                KillSwitchStateRow(
                    set_at=dt.datetime.now(dt.UTC),
                    state="halted",
                    reason="boot:hard_shutdown",
                    set_by="system",
                )
            )
        except Exception as e:  # noqa: BLE001
            log.error(
                "agent.hard_shutdown_halt_insert_failed",
                extra={
                    "event": "agent.hard_shutdown_halt_insert_failed",
                    "error": str(e),
                },
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PendingExecution:
    """Sentinel for AC-7 / AC-12 recovery: replay execution stage only."""

    __slots__ = ("filing", "proposal_id")

    def __init__(self, filing: FilingRow, proposal_id: int) -> None:
        self.filing = filing
        self.proposal_id = proposal_id


def _pending_entry_spend(db_path: str, pending_buys: list[OpenOrder]) -> float:
    """Hotfix 2026-07-08 — dollar value of committed-but-unfilled entry buys.

    Broker-reported cash does not decrease while an entry buy is queued
    (pre-market especially: bracket entries are extended_hours=False and
    sit until the bell), so KS-7 must subtract the spend those orders will
    realize when they fill. Each pending buy is priced from its journal
    `orders` row: qty × limit_price for limit entries, qty ×
    pre_submission_last (the NBBO captured at submission) for market
    entries. Orders with no journal row or no recorded price (e.g. an
    operator-manual order) are skipped with a WARNING — the same blind
    spot as pre-fix, but now visible. A partially-filled order slightly
    over-subtracts (broker cash already reflects the filled part) —
    conservative in the safe direction.
    """
    from journal.repo import get_order_by_broker_id

    total = 0.0
    for o in pending_buys:
        row = get_order_by_broker_id(db_path, o.broker_order_id)
        ref: float | None = None
        if row is not None:
            ref = (
                row.limit_price
                if row.limit_price is not None
                else row.pre_submission_last
            )
        if ref is None:
            log.warning(
                "agent.pending_entry_spend_unpriced",
                extra={
                    "event": "agent.pending_entry_spend_unpriced",
                    "symbol": o.symbol,
                    "broker_order_id": o.broker_order_id,
                },
            )
            continue
        total += o.qty * ref
    return total


class _UniverseAdapter:
    """Adapt `Universe.is_in_universe` to the validator's `_UniverseLike`
    Protocol. (S6.2's validator names the method `is_in_universe(ticker)`.)
    """

    def __init__(self, universe: Any) -> None:
        self._u = universe

    def is_in_universe(self, ticker: str) -> bool:
        return self._u.is_in_universe(ticker)


# Trade-plan fields an Opus `modify` review may overlay onto the working
# copy of the trade payload. `size_pct_of_capital` is deliberately absent
# — it flows through S5.4's working-copy `size_pct_requested` path, not
# through the payload overlay.
_REVIEWER_OVERLAY_FIELDS = (
    "exit_conditions",
    "stop_loss_price",
    "take_profit_price",
)


def _reviewer_trade_overlay(review: Any) -> dict[str, Any]:
    """Map a modify-decision Opus review's journaled modifications
    (`proposal_reviews.modifications_json`) onto `TradeProposal` field
    updates for the in-memory working copy of the trade.

    Returns `{}` when there is no review, the decision isn't `modify`,
    or the modifications carry none of the overlayable fields. Raises on
    a corrupt `modifications_json` — the caller's payload-parse guard
    turns that into the existing `schema` rejection (fail closed: we
    never execute a modified proposal at its unmodified levels).
    """
    if review is None or review.decision != "modify":
        return {}
    if not review.modifications_json:
        return {}
    mods = json.loads(review.modifications_json)
    return {
        field: mods[field]
        for field in _REVIEWER_OVERLAY_FIELDS
        if mods.get(field) is not None
    }


def _stub_execution_config() -> Any:
    """Fallback config used only when `AgentLoop` is constructed without
    a real `AppConfig` (test convenience). Mirrors the v1 defaults from
    `quinn.example.toml`."""
    from config.loader import ExecutionConfig

    return ExecutionConfig(
        broker_mode="paper",
        ks4_pct_cap=0.20,
        ks4_absolute_cap_usd=1000.0,
        ks5_max_concurrent=5,
        ks7_cash_reserve_pct=0.05,
        sizing_mid_pct=0.05,
        sizing_high_pct=0.10,
    )


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


__all__ = ["AgentLoop"]
