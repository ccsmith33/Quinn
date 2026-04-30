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
        if has_evaluator or has_watcher:
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
        if evaluator is None and watcher is None:
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
        # Skip if executions row already exists (idempotency).
        with connect(self._components.journal.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM executions WHERE proposal_id = ?", (proposal_id,)
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

    async def _execute(self, proposal_id: int, filing: FilingRow) -> None:
        """Run the validator → sizer → submitter chain for `proposal_id`.

        Writes an `executions` row in every case (accepted, rejected,
        kill-switch). On reject paths, the agent loop owns the journal
        write because S6.2 / S6.3 are pure logic; on accept, S6.4's
        `OrderSubmitter.submit` writes the row.
        """
        # Idempotency: skip if already executed.
        with connect(self._components.journal.db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM executions WHERE proposal_id = ?", (proposal_id,)
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

        # Build TradeProposal pydantic model from row payload (raw_response).
        from proposal.schemas import TradeProposal, validate_trade_proposal

        try:
            payload = json.loads(proposal_row.raw_response)
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
            self._write_rejected_execution(proposal_id, "schema")
            return

        # If Opus modified the proposal, apply the overlay before sizing.
        # (S5.4 already wrote the working_proposal modifications onto a
        # ProposalRow copy in-memory; we don't re-derive it here, we just
        # honor the size_pct in `proposal_row.size_pct_requested`.)

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
            self._write_rejected_execution(proposal_id, validation.reason)
            return

        # Sizer.
        account = self._components.broker.get_account()
        positions = self._components.broker.get_positions()
        quote = self._components.broker.get_quote(trade.symbol)
        sizing = self._components.sizer.size(
            trade,
            account,
            positions,
            quote,
            self._config.execution if self._config else _stub_execution_config(),
        )
        if isinstance(sizing, SizingRejected):
            self._write_rejected_execution(proposal_id, sizing.reason)
            return

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
            return

        try:
            result = self._components.submitter.submit(
                accepted_proposal,
                self._components.broker,
                self._components.journal,
                self._components.killswitch,
            )
        except BrokerUnavailable as e:
            log.error(
                "agent.broker_unavailable",
                extra={
                    "event": "agent.broker_unavailable",
                    "proposal_id": proposal_id,
                    "error": str(e),
                },
            )
            # The submitter writes a `submission_failed` row on broker
            # outage; if the exception escaped that path entirely, still
            # record so the auditor sees it.
            with connect(self._components.journal.db_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM executions WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
            if row is None:
                self._write_rejected_execution(proposal_id, "broker_unavailable")
            return

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

    # -- Helpers -----------------------------------------------------------

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
        can look up the persisted proposal after analyze() returns."""
        from analyzer.sonnet import compute_decision_id

        sonnet_model_id = self._components.analyzer._sonnet_model_id  # noqa: SLF001
        prompt_version = self._components.analyzer._builder.prompt_version(  # noqa: SLF001
            "sonnet_filing_analysis_v1"
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


class _UniverseAdapter:
    """Adapt `Universe.is_in_universe` to the validator's `_UniverseLike`
    Protocol. (S6.2's validator names the method `is_in_universe(ticker)`.)
    """

    def __init__(self, universe: Any) -> None:
        self._u = universe

    def is_in_universe(self, ticker: str) -> bool:
        return self._u.is_in_universe(ticker)


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
