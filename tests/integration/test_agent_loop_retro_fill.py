"""Feature C — opportunistic retro-fill on slot-open.

Tests `RetroFillCoordinator` end-to-end against the same realistic
component set as the rest of the integration suite (real validator,
real sizer, real submitter, real OpusReviewer; only Anthropic + broker
are faked).

Acceptance criteria covered:

- Slot-open + cv≥9 candidate within 6h → Opus reviews → on accept,
  position opens.
- cv8 candidate: not considered.
- cv9 candidate from 7h ago: not considered.
- Two cv9 candidates: highest-conviction-most-recent picked first.
- Pyramiding guard: candidate symbol already held → coordinator skips.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    FilingRow,
    PositionRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    connect,
    get_proposal_review_by_proposal_id,
    insert_filing,
    insert_position,
    insert_prompt,
    insert_proposal,
)


def _seed_prompt(db: str) -> str:
    pv = "sonnet_filing_analysis@retrotest0001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="x" * 64,
        ),
    )
    return pv


def _seed_opus_prompt(db: str, prompt_builder: Any) -> str:
    """Register the real Opus prompt version so proposal_reviews FK holds."""
    from prompts.loader import ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT

    pv = prompt_builder.prompt_version(ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT)
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name=ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT,
            file_path=f"src/prompts/{ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT}.txt",
            content_hash="y" * 64,
        ),
    )
    return pv


def _insert_pending_proposal(
    db: str,
    *,
    prompt_version: str,
    decision_id: str,
    symbol: str,
    conviction: int,
    raw_text_dir: Path,
    accession: str,
    created_at: dt.datetime,
    cik: int = 320193,
) -> tuple[int, FilingRow]:
    """Insert a proposal that simulates Feature B's "skipped at capacity"
    state: a `trade_proposal` row with no `proposal_reviews` plus a
    placeholder `executions` row whose `reject_reason='pending_capacity'`
    (the deterministic signal `find_retro_candidate` keys on, written
    by `AgentLoop._execute` when Feature B's gate fired upstream).
    Backdates `created_at` so freshness-window tests work.
    """
    raw_path = raw_text_dir / f"{accession}.txt"
    raw_path.write_text(
        f"# Filing {accession}\nIssuer: {symbol}\n"
        "Material acquisition announced; integration timeline plausible.\n"
    )
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=accession,
            cik=cik,
            form_type="8-K",
            filed_at=dt.datetime(2026, 5, 6, 9, 30, 0),
            fetched_at=dt.datetime(2026, 5, 6, 9, 31, 0),
            raw_text_path=str(raw_path),
            content_hash="hash-" + accession,
            item_codes='["1.01"]',
            issuer_ticker=symbol,
        ),
    )
    payload = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": 90.00,
        "time_horizon_days": 14,
        "conviction": conviction,
        "thesis": (
            f"{symbol} announced a material acquisition in 8-K Item 1.01 "
            "with concrete pricing terms; integration timeline is plausible."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 trading days"],
        "risk_factors": ["Closing conditions not yet met"],
    }
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=decision_id,
            model_id="claude-sonnet-4-6",
            prompt_version=prompt_version,
            raw_response=json.dumps(payload),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=conviction,
            thesis=payload["thesis"],
            input_tokens=1500,
            output_tokens=800,
            cache_read_tokens=4000,
            cache_creation_tokens=200,
            latency_ms=2000,
            cost_usd=0.02,
        ),
    )
    # Backdate created_at — required to test the freshness window.
    with connect(db) as conn:
        conn.execute(
            "UPDATE proposals SET created_at = ? WHERE id = ?",
            (created_at, pid),
        )
        conn.commit()

    # Seed the pending_capacity placeholder execution row that Feature
    # B's loop guard would have written upstream.
    from journal.models import ExecutionRow
    from journal.repo import insert_execution
    insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="rejected",
            reject_reason="pending_capacity",
            submitted_orders_json="[]",
        ),
    )

    from journal.repo import get_filing_by_id
    f = get_filing_by_id(db, fid)
    assert f is not None
    return pid, f


def _opus_ratify_json() -> str:
    return json.dumps(
        {
            "decision": "ratify",
            "rationale": (
                "Filing language supports the thesis with a concrete "
                "catalyst and plausible integration timeline. The proposal "
                "sizing is conservative."
            ),
        }
    )


@pytest.mark.asyncio
async def test_retro_fill_picks_cv9_in_window_and_executes(
    db_path: str,
    journal,
    universe,
    prompt_builder,
    proposal_store,
    fake_broker,
    fake_anthropic,
    killswitch,
    tmp_path: Path,
) -> None:
    """Slot-open + cv9 candidate within 6h + symbol not held → Opus
    reviews → ratifies → validator/sizer/submitter chain runs → broker
    sees an entry + stop pair."""
    from analyzer.opus import OpusReviewer
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator
    from execution.orders import OrderSubmitter
    from execution.sizing import SizingEngine
    from execution.validator import ProposalValidator

    register_composed_prompt_versions(prompt_builder, db_path)

    # Universe fixture seeds ACME (CIK 320193) — symbol is in-universe.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sonnet_pv = prompt_builder.prompt_version("sonnet_filing_analysis_v1")
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    pid, _filing = _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-d1",
        symbol="ACME",
        conviction=9,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100001",
        created_at=now - dt.timedelta(hours=2),
    )

    # OpusReviewer is the real one — needs the registered opus pv.
    fake_anthropic.responses_by_purpose = {"review": [_opus_ratify_json()]}

    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )

    # Build the executor closure. Mirrors what AgentLoop._execute does
    # from the loop's POV: validator → sizer → submitter, with rejected
    # execution writes on failure paths.
    validator = ProposalValidator()
    sizer = SizingEngine()
    submitter = OrderSubmitter()

    executor = _make_executor(
        journal=journal,
        validator=validator,
        sizer=sizer,
        submitter=submitter,
        broker=fake_broker,
        killswitch=killswitch,
        universe=universe,
    )

    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
    )
    coordinator.set_executor(executor)

    await coordinator.run_tick()

    # Opus review row exists and is `ratify`.
    review = get_proposal_review_by_proposal_id(db_path, pid)
    assert review is not None
    assert review.decision == "ratify"

    # Two execution rows: the seeded pending_capacity placeholder, then
    # the retro path's accepted outcome (migration 003 allows multiple
    # rows per proposal so the retro insert doesn't collide on UNIQUE).
    with connect(db_path) as conn:
        executions = conn.execute(
            "SELECT decision, reject_reason FROM executions "
            "WHERE proposal_id = ? ORDER BY id ASC",
            (pid,),
        ).fetchall()
    assert len(executions) == 2
    assert executions[0]["reject_reason"] == "pending_capacity"
    assert executions[1]["decision"] == "accepted"
    assert executions[1]["reject_reason"] is None

    # Broker side — entry + stop = 2 submissions.
    sides = [(o.side, o.order_type) for o in fake_broker.submitted_orders]
    assert ("buy", "market") in sides
    assert ("sell", "stop") in sides


@pytest.mark.asyncio
async def test_retro_fill_excludes_cv8(
    db_path: str,
    journal,
    universe,
    prompt_builder,
    proposal_store,
    fake_broker,
    fake_anthropic,
    killswitch,
    tmp_path: Path,
) -> None:
    """cv8 candidate must NOT trigger retro fill (filter at conviction>=9)."""
    from analyzer.opus import OpusReviewer
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator

    register_composed_prompt_versions(prompt_builder, db_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sonnet_pv = prompt_builder.prompt_version("sonnet_filing_analysis_v1")
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-cv8",
        symbol="ACME",
        conviction=8,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100002",
        created_at=now - dt.timedelta(hours=1),
    )

    fake_anthropic.responses_by_purpose = {"review": []}
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )
    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
    )

    called: list[int] = []

    async def _executor(pid: int, filing: FilingRow) -> None:
        called.append(pid)

    coordinator.set_executor(_executor)
    await coordinator.run_tick()

    # No Opus call, no executor call.
    assert fake_anthropic.calls == []
    assert called == []


@pytest.mark.asyncio
async def test_retro_fill_excludes_stale_window(
    db_path: str,
    journal,
    universe,
    prompt_builder,
    proposal_store,
    fake_broker,
    fake_anthropic,
    killswitch,
    tmp_path: Path,
) -> None:
    """cv9 candidate from 7 hours ago must NOT trigger retro fill."""
    from analyzer.opus import OpusReviewer
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator

    register_composed_prompt_versions(prompt_builder, db_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sonnet_pv = prompt_builder.prompt_version("sonnet_filing_analysis_v1")
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-stale",
        symbol="ACME",
        conviction=9,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100003",
        created_at=now - dt.timedelta(hours=7),
    )

    fake_anthropic.responses_by_purpose = {"review": []}
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )
    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
    )

    called: list[int] = []

    async def _executor(pid: int, filing: FilingRow) -> None:
        called.append(pid)

    coordinator.set_executor(_executor)
    await coordinator.run_tick()

    assert fake_anthropic.calls == []
    assert called == []


@pytest.mark.asyncio
async def test_retro_fill_picks_higher_conviction_first(
    db_path: str,
    journal,
    universe,
    prompt_builder,
    proposal_store,
    fake_broker,
    fake_anthropic,
    killswitch,
    tmp_path: Path,
) -> None:
    """Two cv9+ candidates: highest conviction (cv10) wins. We assert by
    examining which decision_id Opus was called against."""
    from analyzer.opus import OpusReviewer
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator

    register_composed_prompt_versions(prompt_builder, db_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sonnet_pv = prompt_builder.prompt_version("sonnet_filing_analysis_v1")
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    pid_cv9, _ = _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-cv9",
        symbol="ACME",
        conviction=9,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100004",
        created_at=now - dt.timedelta(hours=1),
    )
    pid_cv10, _ = _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-cv10",
        symbol="WIDG",
        conviction=10,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100005",
        created_at=now - dt.timedelta(hours=2),
        cik=789019,  # WIDG is also seeded in the universe fixture
    )

    fake_anthropic.responses_by_purpose = {"review": [_opus_ratify_json()]}

    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )

    called: list[int] = []

    async def _executor(pid: int, filing: FilingRow) -> None:
        called.append(pid)

    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
    )
    coordinator.set_executor(_executor)
    await coordinator.run_tick()

    # cv10 picked first.
    assert called == [pid_cv10]
    assert pid_cv9 not in called
    # Opus review row written for cv10 only.
    review_cv10 = get_proposal_review_by_proposal_id(db_path, pid_cv10)
    review_cv9 = get_proposal_review_by_proposal_id(db_path, pid_cv9)
    assert review_cv10 is not None
    assert review_cv9 is None


@pytest.mark.asyncio
async def test_retro_fill_pyramiding_guard_skips_held_symbol(
    db_path: str,
    journal,
    universe,
    prompt_builder,
    proposal_store,
    fake_broker,
    fake_anthropic,
    killswitch,
    tmp_path: Path,
) -> None:
    """CRITICAL: if a candidate's symbol is already in an open position,
    the coordinator MUST skip — no Opus call, no execute. Reviewer will
    block on this missing."""
    from analyzer.opus import OpusReviewer
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator

    register_composed_prompt_versions(prompt_builder, db_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sonnet_pv = prompt_builder.prompt_version("sonnet_filing_analysis_v1")
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    pid, _ = _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-pyramid",
        symbol="ACME",
        conviction=9,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100006",
        created_at=now - dt.timedelta(hours=1),
    )

    # Pre-seed an open position in ACME.
    insert_position(
        db_path,
        PositionRow(
            snapshot_at=now - dt.timedelta(minutes=10),
            source="reconciler",
            symbol="ACME",
            qty=50,
            avg_entry_price=100.0,
            market_value=5000.0,
            unrealized_pnl=0.0,
        ),
    )

    fake_anthropic.responses_by_purpose = {"review": []}
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )

    called: list[int] = []

    async def _executor(p: int, filing: FilingRow) -> None:
        called.append(p)

    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
    )
    coordinator.set_executor(_executor)
    await coordinator.run_tick()

    # No Opus call (we skip BEFORE Opus to avoid spend on a guaranteed-bad path).
    assert fake_anthropic.calls == []
    assert called == []
    # No proposal_review row written.
    review = get_proposal_review_by_proposal_id(db_path, pid)
    assert review is None


@pytest.mark.asyncio
async def test_retro_fill_skips_when_no_slack(
    db_path: str,
    journal,
    universe,
    prompt_builder,
    proposal_store,
    fake_broker,
    fake_anthropic,
    killswitch,
    tmp_path: Path,
) -> None:
    """If KS5 is at cap (no slack), coordinator returns silently — no
    Opus call, no executor call."""
    from analyzer.opus import OpusReviewer
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator

    register_composed_prompt_versions(prompt_builder, db_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sonnet_pv = prompt_builder.prompt_version("sonnet_filing_analysis_v1")
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-fullcap",
        symbol="ACME",
        conviction=9,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100007",
        created_at=now - dt.timedelta(hours=1),
    )

    # Seed 5 different open positions to fill KS5 capacity. Symbols
    # other than ACME/WIDG so the candidate isn't held; we want the
    # capacity gate to be the rejection reason, not pyramiding.
    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        insert_position(
            db_path,
            PositionRow(
                snapshot_at=now - dt.timedelta(minutes=10 + i),
                source="reconciler",
                symbol=sym,
                qty=10,
                avg_entry_price=50.0,
                market_value=500.0,
                unrealized_pnl=0.0,
            ),
        )

    fake_anthropic.responses_by_purpose = {"review": []}
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )

    called: list[int] = []

    async def _executor(p: int, filing: FilingRow) -> None:
        called.append(p)

    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
    )
    coordinator.set_executor(_executor)
    await coordinator.run_tick()

    assert fake_anthropic.calls == []
    assert called == []


@pytest.mark.asyncio
async def test_retro_review_input_is_capped_like_day_zero(
    db_path: str,
    journal,
    prompt_builder,
    proposal_store,
    fake_anthropic,
    tmp_path: Path,
) -> None:
    """Advisory #5 — retro-fill's Opus review sees the SAME capped filing
    body a day-0 analysis would have (shared `cap_filing_text` helper:
    cap applied, marker appended, over-cap tail dropped). Without this,
    every retro review shipped the full untruncated filing and the cap's
    savings silently stopped applying off the day-0 path.

    End-to-end through the REAL OpusReviewer + PromptBuilder: the FULL
    capped body must reach the Opus request's block 3, not a 2,000-char
    cover-page slice (evidence-symmetry fix)."""
    from analyzer.opus import OpusReviewer
    from analyzer.sonnet import cap_filing_text, truncation_marker
    from app.composition import register_composed_prompt_versions
    from app.retro_fill import RetroFillCoordinator

    register_composed_prompt_versions(prompt_builder, db_path)

    raw_dir = tmp_path / "raw-cap"
    raw_dir.mkdir()
    sonnet_pv = _seed_prompt(db_path)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _pid, filing = _insert_pending_proposal(
        db_path,
        prompt_version=sonnet_pv,
        decision_id="retro-cap1",
        symbol="ACME",
        conviction=9,
        raw_text_dir=raw_dir,
        accession="0001234567-26-100077",
        created_at=now - dt.timedelta(hours=2),
    )
    cap = 20_000
    raw_body = ("material 8-K item section line\n" * 1_500) + "TAIL_SENTINEL"
    Path(filing.raw_text_path).write_text(raw_body, encoding="utf-8")

    fake_anthropic.responses_by_purpose = {"review": [_opus_ratify_json()]}
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
        max_input_chars=cap,
    )

    coordinator = RetroFillCoordinator(
        journal=journal,
        opus_reviewer=reviewer,
        ks5_max_concurrent=5,
        now_fn=lambda: now,
        max_input_chars=cap,
    )

    async def _noop_executor(proposal_id: int, f: FilingRow) -> None:
        return None

    coordinator.set_executor(_noop_executor)
    await coordinator.run_tick()

    # The Opus request carries the capped body — whole, once, marker intact.
    review_calls = [c for c in fake_anthropic.calls if c["purpose"] == "review"]
    assert len(review_calls) == 1
    block3 = "".join(
        b.text for m in review_calls[0]["request"].messages for b in m.content
    )
    expected_capped = cap_filing_text(raw_body, cap=cap, filing_id=filing.id)
    marker = truncation_marker(cap)
    assert expected_capped.endswith(marker)
    assert len(expected_capped) <= cap + len(marker)
    assert expected_capped in block3          # FULL capped body, end-to-end
    assert "TAIL_SENTINEL" not in block3      # over-cap tail dropped
    assert block3.count("[NOTE: Filing truncated") == 1  # no re-truncation


# ---------------------------------------------------------------------------
# Helper: a minimal executor mirroring AgentLoop._execute's chain.
#
# We don't construct the full AgentLoop because (a) the loop owns an
# asyncio queue + RSS pump we don't need here, and (b) the retro path
# only needs validator → sizer → submitter + the Opus-decision check.
# This mirror is intentional: the integration test pins the contract
# RetroFillCoordinator depends on without coupling to AgentLoop's
# internals.
# ---------------------------------------------------------------------------


def _make_executor(
    *,
    journal: Any,
    validator: Any,
    sizer: Any,
    submitter: Any,
    broker: Any,
    killswitch: Any,
    universe: Any,
):
    from execution.orders import AcceptedProposal
    from execution.sizing import SizingRejected
    from execution.validator import Rejected
    from journal.models import ExecutionRow
    from journal.repo import (
        connect,
        get_proposal_by_id,
        get_proposal_review_by_proposal_id,
    )
    from proposal.schemas import validate_trade_proposal

    class _UniverseAdapter:
        def __init__(self, u: Any) -> None:
            self._u = u

        def is_in_universe(self, ticker: str) -> bool:
            return self._u.is_in_universe(ticker)

    def _stub_execution_config() -> Any:
        from config.loader import ExecutionConfig
        return ExecutionConfig(
            broker_mode="paper",
            ks4_pct_cap=0.20,
            ks4_absolute_cap_usd=100_000.0,
            ks5_max_concurrent=5,
            ks7_cash_reserve_pct=0.05,
            sizing_mid_pct=0.05,
            sizing_high_pct=0.10,
        )

    def _write_rejected(proposal_id: int, reason: str) -> None:
        journal.insert_execution(
            ExecutionRow(
                proposal_id=proposal_id,
                decision="rejected",
                reject_reason=reason,
                submitted_orders_json="[]",
            )
        )

    async def _execute(proposal_id: int, filing: FilingRow) -> None:
        # Mirror the loop's idempotency: pending_capacity placeholders
        # don't count as terminal outcomes.
        with connect(journal.db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM executions "
                "WHERE proposal_id = ? "
                "AND (reject_reason IS NULL OR reject_reason != 'pending_capacity') "
                "LIMIT 1",
                (proposal_id,),
            ).fetchone()
        if existing is not None:
            return

        proposal_row = get_proposal_by_id(journal.db_path, proposal_id)
        if proposal_row is None:
            return
        review = get_proposal_review_by_proposal_id(journal.db_path, proposal_id)
        if review is not None and review.decision in ("reject", "malformed"):
            _write_rejected(proposal_id, "opus_reject")
            return

        try:
            payload = json.loads(proposal_row.raw_response)
            trade = validate_trade_proposal(payload)
        except Exception:  # noqa: BLE001
            _write_rejected(proposal_id, "schema")
            return

        validation = validator.validate(
            trade, broker, _UniverseAdapter(universe), killswitch, journal
        )
        if isinstance(validation, Rejected):
            _write_rejected(proposal_id, validation.reason)
            return

        account = broker.get_account()
        positions = broker.get_positions()
        quote = broker.get_quote(trade.symbol)
        sizing = sizer.size(
            trade, account, positions, quote, _stub_execution_config()
        )
        if isinstance(sizing, SizingRejected):
            _write_rejected(proposal_id, sizing.reason)
            return

        ap = AcceptedProposal(
            proposal=trade,
            proposal_id=proposal_id,
            qty=sizing.qty,
            realized_dollar_size=sizing.realized_dollar_size,
            realized_pct=sizing.realized_pct,
            realized_dollar_size_request=sizing.realized_dollar_size_request,
        )
        submitter.submit(ap, broker, journal, killswitch)

    return _execute
