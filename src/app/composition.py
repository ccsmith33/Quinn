"""S5.6 AC-2 — agent-loop dependency wiring.

`compose_agent` is the one place where every E2..E7 component is
constructed and stitched together. Single code path for paper / live
(D-007 sacred): broker_mode picks the credential payload, nothing else.

Order of construction matters: the journal must exist first; the
universe must already have a snapshot (load_latest raises otherwise);
the kill-switch wraps the journal; the broker is built last because
its credentials cross from `Secrets` into a live SDK client.

Carry-forward S5.2 D-1 (medium): the composed prompt-version ids
emitted by `PromptBuilder.prompt_version(...)` must be registered in
the `prompts` table BEFORE the analyzer makes its first API call,
otherwise the `proposals.prompt_version` FK fires AFTER the spend.
This is `register_composed_prompt_versions(...)`. Reviewer-e5 will
scrutinize the call site.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from analyzer.anthropic_client import AnthropicClient
from analyzer.opus import OpusReviewer
from analyzer.sonnet import SonnetAnalyzer
from analyzer.thesis_review import ThesisReviewer
from app.retro_fill import RetroFillCoordinator
from app.thesis_coordinator import (
    ThesisReviewCoordinator,
    make_journal_filings_lookup,
)
from broker.alpaca import AlpacaBroker
from broker.protocol import BrokerAdapter
from config.loader import AppConfig
from config.secrets import Secrets
from execution.exit_policy import ExitPolicyTicker
from execution.orders import OrderSubmitter
from execution.pdt_budget import PDTState  # PDT-SUNSET-2026-06-04: ADR-009 wiring.
from execution.pdt_transition import (  # PDT-TRANSITION-D-077: survives P2.
    ConverterBroker,
    PDTTransitionConverter,
)
from execution.sizing import SizingEngine
from execution.validator import ProposalValidator
from execution.virtual_exits import (  # PDT-SUNSET-2026-06-04
    DeferredSellReplayer,
    VirtualExitScanner,
)
from ingestion.detail_fetcher import DetailFetcher
from ingestion.edgar_client import EdgarClient
from ingestion.rss_loop import DiscoveredFiling, RssDiscoveryLoop
from ingestion.ticker_resolver import TickerResolver
from journal.models import FilingRow, PromptRow
from journal.repo import JournalRepo, insert_prompt
from killswitch.api import KillSwitch
from killswitch.auto_halts import AutoHaltEvaluator
from observability.alerts import AlertWatcher, TelegramNotifier
from observability.log_port import get_logger
from prefilter.orchestrator import Prefilter
from prefilter.similarity import SimilarityChecker
from prompts.loader import PromptBuilder
from proposal.store import ProposalStore
from reconciler.fill_ingest import FillIngestor
from reconciler.reconciler import Reconciler
from universe.api import Universe

log = get_logger(__name__)

# Names whose composed pv must be registered before any LLM call. Maps
# 1:1 to `_PROMPT_DEFS` in `prompts.loader`. The D-079 v2s are
# registered alongside the v1s: the Opus stages run v2 now, the sonnet
# stage flips at integration (see ACTIVE_SONNET_ANALYSIS_PROMPT in
# prompts.loader) — pre-registering makes the flip journal-safe.
_COMPOSED_PROMPT_NAMES: tuple[str, ...] = (
    "sonnet_filing_analysis_v1",
    "sonnet_filing_analysis_v2",
    "opus_proposal_review_v1",
    "opus_proposal_review_v2",
    "opus_thesis_review_v1",
    "opus_thesis_review_v2",
)


@dataclass(frozen=True)
class AgentComponents:
    """Bundle of every dependency the agent loop coordinates.

    The story's dev-notes list `execution: ExecutionLayer` as one field;
    the actual surface in v1 is the validator + sizer + submitter trio
    plus the broker. They're listed individually here so the loop can
    call them in the order the story prescribes (validate → size →
    submit). `execution` is kept as `None`-able for forward shape
    parity with the story spec.
    """

    universe: Universe
    prefilter: Prefilter
    analyzer: SonnetAnalyzer
    reviewer: OpusReviewer
    proposal_store: ProposalStore
    validator: ProposalValidator
    sizer: SizingEngine
    submitter: OrderSubmitter
    execution: object | None  # placeholder per story dev-notes shape
    broker: BrokerAdapter
    reconciler: Any  # Reconciler in prod; stub in tests (start/stop)
    killswitch: KillSwitch
    ingestion_queue: asyncio.Queue[FilingRow]
    journal: JournalRepo
    auto_halt_evaluator: AutoHaltEvaluator | None = None
    # S5.6 carry-fwd S8.2 wiring: AlertWatcher polled on the 60s tick.
    alert_watcher: AlertWatcher | None = None
    # RSS-path wiring (production-bug fix): the discovery loop pushes
    # `DiscoveredFiling` onto `discovered_queue`; a pump task drains it
    # through `detail_fetcher` and forwards the resulting `FilingRow`
    # onto `ingestion_queue` for the consumer. All three are optional
    # so unit tests that exercise the consumer in isolation still work.
    rss_discovery_loop: RssDiscoveryLoop | None = None
    detail_fetcher: DetailFetcher | None = None
    discovered_queue: asyncio.Queue[DiscoveredFiling] | None = None
    # PDT-SUNSET-2026-06-04: ADR-009 — process-shared activation flag.
    # Held on the components bundle so the agent loop can pass it to
    # the OrderSubmitter (S-PDT-3) and the reconciler can refresh it.
    pdt_state: PDTState | None = None
    # PDT-SUNSET-2026-06-04: ADR-009 §"Pre-market deferred replay" —
    # invoked once per session at boot by `AgentLoop._boot`.
    deferred_replayer: Any | None = None
    # PDT-TRANSITION-D-077 (ADR-012 §4.2): one-shot boot-time converter
    # of active virtual exits to GTC broker orders; runs before the
    # deferred replayer. Survives the P2 tag-removal PR; removed in the
    # post-soak cleanup.
    pdt_transition_converter: Any | None = None


def compose_agent(
    config: AppConfig,
    *,
    secrets: Secrets,
    journal: JournalRepo,
    prompts_dir: Path,
    similarity_artifact_dir: str,
) -> Any:
    """Construct the live agent. Returns an `AgentLoop`.

    `prompts_dir` and `similarity_artifact_dir` are path-shaped values
    plumbed from `src/app.py` (which reads them from runtime defaults
    or env). Keeping them as parameters makes this function unit-
    testable without monkey-patching the filesystem.

    Single code path for paper / live (D-007): the only place we look
    at `config.execution.broker_mode` is when constructing the broker
    — to pick which credential pair Alpaca expects and which endpoint
    to hit. No runtime branch reads it.
    """
    # Universe lookup (S2.3) — fail loud if no snapshot exists; the
    # agent must not boot without a universe (story §"Crash recovery").
    universe = Universe.load_latest(journal.db_path)

    # Prefilter (S4.3) — the similarity checker depends on the journal
    # path + an artifact dir for vectorizers + threshold from config.
    similarity = SimilarityChecker(
        db_path=journal.db_path,
        artifact_dir=similarity_artifact_dir,
        threshold=config.prefilter.similarity_threshold,
    )
    prefilter = Prefilter(
        db_path=journal.db_path,
        universe=universe,
        similarity=similarity,
        form_4_enabled=config.prefilter.form_4_enabled,
    )

    # Anthropic client (S5.2) — single instance shared by analyzer +
    # reviewer; purpose tag distinguishes their llm_calls rows.
    anthropic = AnthropicClient(
        api_key=secrets.anthropic_api_key,
        default_model_id=config.analyzer.sonnet_model_id,
        db_path=journal.db_path,
    )

    # Prompt registry → register composed pvs BEFORE first call.
    prompt_builder = PromptBuilder(prompts_dir)
    register_composed_prompt_versions(prompt_builder, journal.db_path)

    # Proposal store (S5.5).
    proposal_store = ProposalStore(db_path=journal.db_path)

    # Opus reviewer (S5.4) — built first so the analyzer can hold it.
    # `opus_max_output_tokens` plumb-through is the S5.6 carry-fwd
    # resolution for S5.2 reviewer C-1 (D-052).
    reviewer = OpusReviewer(
        client=anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id=config.analyzer.opus_model_id,
        ks4_pct_cap=config.execution.ks4_pct_cap,
        db_path=journal.db_path,
        max_output_tokens=config.analyzer.opus_max_output_tokens,
    )

    # Broker — the SOLE credential seam. AlpacaBroker validates
    # endpoint↔mode at construction time, so a misconfigured pair
    # raises here, before any trading happens. Constructed BEFORE the
    # analyzer (Feature B) because the analyzer's KS-5 capacity counter
    # must consult broker truth (same source-of-truth as
    # `SizingEngine.size()` so the two gates can't disagree).
    broker = AlpacaBroker(
        mode=config.execution.broker_mode,
        api_key_id=secrets.alpaca_api_key_id,
        api_secret=secrets.alpaca_api_secret_key,
        endpoint=secrets.alpaca_endpoint.get_secret_value(),
    )

    # Sonnet analyzer (S5.3) — the conviction-threshold gate to Opus
    # lives in this constructor (D-047). Feature B: also wires the KS5
    # capacity gate so we skip Opus spend when no slot is free.
    #
    # Feature B source-of-truth: the capacity counter calls the BROKER
    # (not the journal) so it agrees with `SizingEngine.size()` which
    # also consults `broker.get_positions()`. Mismatch would cause
    # Feature B to refuse Opus spend on a trade that would actually
    # have fit at the sizer's gate.
    analyzer = SonnetAnalyzer(
        client=anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_reviewer=reviewer,
        sonnet_model_id=config.analyzer.sonnet_model_id,
        opus_review_conviction_threshold=(
            config.analyzer.opus_review_conviction_threshold
        ),
        db_path=journal.db_path,
        max_output_tokens=config.analyzer.sonnet_max_output_tokens,
        ks5_max_concurrent=config.execution.ks5_max_concurrent,
        open_positions_counter=lambda: len(broker.get_positions()),
    )

    # Execution stack (S6.2 / S6.3 / S6.4) — pure logic, no broker yet.
    # D-079 §3.2: exit-geometry floor wired from config; trade-time
    # price floor likewise (2026-07-17, ABUS-at-$4.79 incident).
    validator = ProposalValidator(
        min_reward_risk=config.execution.min_reward_risk,
        price_floor_usd=config.execution.price_floor_usd,
    )
    sizer = SizingEngine()
    submitter = OrderSubmitter()

    # Kill-switch read API (S7.1). WS1 (D-078, delta §2.3): re-page
    # throttle window for fingerprinted (reconciler) halts from config.
    killswitch = KillSwitch(
        journal, repage_minutes=config.killswitch.halt_repage_minutes
    )
    auto_halt = AutoHaltEvaluator()

    # Telegram outbound notifier. Single instance shared by reconciler
    # (`notify(message)` via _TelegramAlerterAdapter) and AlertWatcher
    # (`send(text)` directly). NFR-9: send is best-effort, never blocks.
    telegram = TelegramNotifier(
        bot_token=secrets.telegram_bot_token.get_secret_value(),
        chat_id=secrets.telegram_operator_chat_id.get_secret_value(),
    )

    # Feature C — opportunistic retro-fill coordinator. Runs after each
    # successful reconciler tick to find high-conviction proposals that
    # were skipped at capacity by Feature B. The executor closure is
    # wired below, AFTER the AgentLoop is constructed (two-phase init).
    retro_coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=config.execution.ks5_max_concurrent,
    )

    # Feature A — thesis-review coordinator. Reviews open positions
    # whose declared horizon has elapsed, then applies hold / close /
    # adjust_stop. Also driven by the reconciler tick.
    thesis_reviewer = ThesisReviewer(
        client=anthropic,
        prompt_builder=prompt_builder,
        opus_model_id=config.analyzer.opus_model_id,
        db_path=journal.db_path,
        max_output_tokens=config.analyzer.opus_max_output_tokens,
    )
    thesis_coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=thesis_reviewer,
        filings_lookup=make_journal_filings_lookup(journal.db_path),
        # Jurisdiction zones: the first trail stage's gain_pct is the
        # Zone-3 line the review context reports (default +20%).
        trail_stages=config.execution.trail_stages,
        # Capacity-pressure sweep (EventReviewsConfig). Off unless the
        # operator sets `[event_reviews] capacity_target_slots` (and
        # daily_sweep); the ExecutionConfig supplies the KS-5
        # effective-cap curve.
        execution_config=config.execution,
        capacity_target_slots=config.event_reviews.capacity_target_slots,
    )

    # PDT-SUNSET-2026-06-04: ADR-009 §3.1 — activation gate. Read once
    # at boot from the broker; refreshed every reconciler tick. The
    # boot-time `get_account()` call is needed regardless of pdt_enabled
    # so PDTState carries the real `last_equity` (the operator escape
    # hatch suppresses the gate, not the read).
    boot_account = broker.get_account()
    pdt_state = PDTState.from_account(
        boot_account, pdt_enabled=config.execution.pdt_enabled
    )
    log.info(
        "pdt.boot",
        extra={
            "event": "pdt.boot",
            "is_active": pdt_state.is_active(),
            "last_equity": boot_account.last_equity,
            "pdt_enabled": config.execution.pdt_enabled,
        },
    )
    # PDT-SUNSET-2026-06-04: ADR-009 §3.2 — virtual exit scanner runs
    # once per reconciler tick. Constructed here so the reconciler can
    # be wired below.
    virtual_exits_scanner = VirtualExitScanner(
        broker=broker, journal=journal, pdt_state=pdt_state,
    )
    # PDT-SUNSET-2026-06-04: ADR-009 §"Pre-market deferred replay" —
    # one-shot replayer invoked at boot from `AgentLoop._boot`. The
    # boot path is the seam (no separate pre-market hook in v1); a
    # deterministic client_order_id + Alpaca's idempotent dedup
    # protects against double-fire on rapid restart.
    deferred_replayer = DeferredSellReplayer(
        broker=broker, journal=journal,
    )
    # PDT-TRANSITION-D-077 (ADR-012 §4.2): boot-time converter, wired
    # ahead of the replayer in `AgentLoop._boot`. The cast bridges the
    # §5 merge order: WS2 adds `submit_oco_sell` to the broker before
    # WS3-P1 merges; until then the converter's per-group isolation
    # degrades an unexpected missing method to a logged error while the
    # scanner keeps covering the affected exits.
    pdt_transition_converter = PDTTransitionConverter(
        broker=cast("ConverterBroker", broker), journal=journal,
    )

    # WS1 (D-078, ADR-010) — poll-based fill ingestion, invoked at the
    # top of every reconcile tick (inside the Reconciler; deliberately
    # NOT a loop.py hook — that is the WS1/WS3 seam resolution).
    fill_ingestor = FillIngestor(broker=broker, journal=journal)

    # D-079 §3.5/§3.6 — exit-policy ticker (trailing ratchet +
    # stale-entry hygiene). Runs once per reconciler tick, after the
    # thesis hook, in the slot the PDT scanner vacates in P2, via the
    # `exit_policy_ticker` ctor seam WS1's reconciler exposes
    # (ADR-011: process down never lowers protection — the broker-side
    # GTC orders remain the protection when the ticker isn't running).
    exit_policy_ticker = ExitPolicyTicker(
        journal=journal,
        broker=broker,
        trail_activation_r=config.execution.trail_activation_r,
        min_ratchet_step_pct=config.execution.min_ratchet_step_pct,
        trail_stages=config.execution.trail_stages,
        breakeven_floor_gain_pct=config.execution.breakeven_floor_gain_pct,
    )

    # Event-triggered thesis reviews ([event_reviews]) — constructed
    # ONLY when the flag is on so a disabled config wires `None` into
    # the reconciler and the whole feature is unreachable (the A/B
    # requirement: byte-identical behavior when disabled). The
    # prev-close source is the daily universe snapshot (same value the
    # analyzer's block-2 summary uses); a held name absent from the
    # universe simply skips the anomaly trigger.
    event_trigger_engine = None
    if config.event_reviews.enabled:
        from app.event_triggers import EventTriggerEngine

        def _prev_close(symbol: str) -> float | None:
            member = universe.get_member(symbol)
            return member.prev_close if member is not None else None

        # News trigger data source — same Alpaca credential pair as the
        # trading adapter (the news API rides the data plan). Best
        # effort: a failed construction degrades to news-off while the
        # filing / anomaly / gain-cross triggers keep running.
        news_client = None
        try:
            from broker.news import AlpacaNewsClient

            news_client = AlpacaNewsClient(
                api_key_id=secrets.alpaca_api_key_id,
                api_secret=secrets.alpaca_api_secret_key,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "event_triggers.news_client_unavailable",
                extra={
                    "event": "event_triggers.news_client_unavailable",
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )

        event_trigger_engine = EventTriggerEngine(
            journal=journal,
            broker=broker,
            enabled=True,
            anomaly_move_pct=config.event_reviews.anomaly_move_pct,
            cooldown_hours=config.event_reviews.cooldown_hours,
            gain_thresholds_pct=config.event_reviews.gain_thresholds_pct,
            trail_stages=config.execution.trail_stages,
            prev_close_lookup=_prev_close,
            news_client=news_client,
            daily_sweep=config.event_reviews.daily_sweep,
            sweep_after_utc_hour=config.event_reviews.sweep_after_utc_hour,
        )
        log.info(
            "event_triggers.enabled",
            extra={
                "event": "event_triggers.enabled",
                "anomaly_move_pct": config.event_reviews.anomaly_move_pct,
                "cooldown_hours": config.event_reviews.cooldown_hours,
                "gain_thresholds_pct": config.event_reviews.gain_thresholds_pct,
                "daily_sweep": config.event_reviews.daily_sweep,
                "sweep_after_utc_hour": config.event_reviews.sweep_after_utc_hour,
            },
        )

    # Reconciler (S6.5) — depends on broker + journal + ks + cfg + alerter.
    # S5.6 carry-fwd S6.5 reviewer M-3 (HIGH): wire the Telegram alerter
    # at construction so position-discrepancy notifications reach the
    # operator (not just structured logs).
    # Feature C: pass the retro coordinator so it fires at end-of-tick.
    # Feature A: pass the thesis coordinator so open-position reviews
    # fire at end-of-tick once their horizon has elapsed.
    reconciler = Reconciler(
        broker=broker,
        journal=journal,
        ks=killswitch,
        cfg=config.reconciler,
        alerter=_TelegramAlerterAdapter(telegram),
        retro_coordinator=retro_coordinator,
        thesis_coordinator=thesis_coordinator,
        # PDT-SUNSET-2026-06-04: refresh the activation flag once per
        # successful reconcile tick. ADR-009 §3.1.
        pdt_state=pdt_state,
        pdt_enabled=config.execution.pdt_enabled,
        virtual_exits_scanner=virtual_exits_scanner,
        # WS1 (D-078): fill truth first, every tick.
        fill_ingestor=fill_ingestor,
        # D-079 §3.5: exit-policy hook, after the thesis hook.
        exit_policy_ticker=exit_policy_ticker,
        # [event_reviews] — None unless the config flag is on.
        event_trigger_engine=event_trigger_engine,
    )

    # AlertWatcher (S8.2) — must be constructed AFTER migrations have
    # run because it snapshots latest `kill_switch_state` to suppress
    # the boot seed row firing a phantom flip. Migrations are run by
    # `main()` BEFORE compose_agent, so we're safe here.
    alert_watcher = AlertWatcher(journal=journal, notifier=telegram)

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()

    # RSS discovery path (S3.1 + S3.2 + S3.3) — the production bug-fix.
    # A single shared `EdgarClient` is reused across discovery, detail
    # fetch, and (via S3.4) the submissions reconciler per ADR-002 §6.
    # The agent loop owns lifecycle: `start()` schedules the RSS poll
    # task, `stop()` cancels it and persists the cursor.
    edgar = EdgarClient(user_agent=config.ingestion.edgar_user_agent)
    discovered_queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    rss_discovery_loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=discovered_queue,
        cursor_path=Path(config.ingestion.rss_cursor_path),
        poll_market_seconds=config.ingestion.rss_poll_seconds_market,
        poll_offhours_seconds=config.ingestion.rss_poll_seconds_offhours,
    )
    # Hotfix 2026-05-04: prose-form ingestion (8-K / 10-Q / 425 / 424B5 /
    # DEF 14A) historically wrote NULL `issuer_ticker`, which made the
    # analyzer treat every issuer as not-in-universe and refuse all
    # proposals. The resolver consults SEC's `company_tickers.json` (with
    # local cache) and is shared via the existing `EdgarClient` so SEC
    # rate-limits and User-Agent headers are honored.
    ticker_resolver = TickerResolver(
        edgar=edgar,
        cache_path=Path(config.ingestion.ticker_cache_path),
    )
    detail_fetcher = DetailFetcher(
        edgar=edgar,
        db_path=journal.db_path,
        raw_root=Path(config.ingestion.raw_filings_root),
        ticker_resolver=ticker_resolver,
    )

    components = AgentComponents(
        universe=universe,
        prefilter=prefilter,
        analyzer=analyzer,
        reviewer=reviewer,
        proposal_store=proposal_store,
        validator=validator,
        sizer=sizer,
        submitter=submitter,
        execution=None,
        broker=broker,
        reconciler=reconciler,
        killswitch=killswitch,
        ingestion_queue=queue,
        journal=journal,
        auto_halt_evaluator=auto_halt,
        alert_watcher=alert_watcher,
        rss_discovery_loop=rss_discovery_loop,
        detail_fetcher=detail_fetcher,
        discovered_queue=discovered_queue,
        # PDT-SUNSET-2026-06-04: ADR-009 wiring.
        pdt_state=pdt_state,
        deferred_replayer=deferred_replayer,
        # PDT-TRANSITION-D-077: ADR-012 §4.2 wiring (survives P2).
        pdt_transition_converter=pdt_transition_converter,
    )

    # Lazy import to keep `app/composition.py` testable without pulling
    # `app/loop.py` (which has heavier asyncio surface).
    from .loop import AgentLoop

    agent_loop = AgentLoop(
        components=components,
        config=config,
        shutdown_grace_seconds=60.0,
    )

    # Feature C two-phase wiring: the retro coordinator was created
    # BEFORE the loop (since the reconciler holds it), so set the
    # executor now that the loop exists. Without this, `run_tick` is a
    # no-op — fail safe.
    retro_coordinator.set_executor(agent_loop.execute_proposal)

    return agent_loop


def register_composed_prompt_versions(
    builder: PromptBuilder, db_path: str
) -> None:
    """Register every composed pv in the `prompts` table.

    Required before the first `AnthropicClient.call` because
    `proposals.prompt_version` is FK-enforced (S5.2 D-1 carry-forward).
    Idempotent: PK collisions resolve to "already registered".
    """
    for name in _COMPOSED_PROMPT_NAMES:
        version = builder.prompt_version(name)
        composed_bytes = builder._composed_system_bytes(name)  # type: ignore[attr-defined]
        import hashlib

        digest = hashlib.sha256(composed_bytes).hexdigest()
        try:
            insert_prompt(
                db_path,
                PromptRow(
                    prompt_version=version,
                    name=name,
                    file_path=str(Path(builder._dir) / f"{name}.txt"),  # type: ignore[attr-defined]
                    content_hash=digest,
                ),
            )
            log.info(
                "composed prompt registered",
                extra={
                    "event": "compose.prompt_registered",
                    "prompt_version": version,
                    "prompt_name": name,
                },
            )
        except sqlite3.IntegrityError:
            log.debug(
                "composed prompt already registered",
                extra={
                    "event": "compose.prompt_already_registered",
                    "prompt_version": version,
                    "prompt_name": name,
                },
            )


class _TelegramAlerterAdapter:
    """Bridge `Reconciler._AlerterLike.notify(message)` to
    `Notifier.send(text)`.

    The reconciler (S6.5) and AlertWatcher (S8.2) settled on different
    method names for the same operation. Rather than churn either, this
    one-line adapter satisfies the reconciler's structural Protocol.
    """

    def __init__(self, notifier: Any) -> None:
        self._notifier = notifier

    def notify(self, message: str) -> None:
        self._notifier.send(message)


__all__ = [
    "AgentComponents",
    "compose_agent",
    "register_composed_prompt_versions",
]
