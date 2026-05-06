"""Shared fixtures for S5.6 agent-loop integration tests.

The integration tests exercise the real `AgentLoop` against real
`Prefilter`, `SonnetAnalyzer`, `OpusReviewer`, `ProposalStore`,
`ProposalValidator`, `SizingEngine`, and `OrderSubmitter` (S4.3 / S5.3 /
S5.4 / S5.5 / S6.2 / S6.3 / S6.4) — the only fakes are at the
*external* boundaries:

  - `_FakeAnthropicClient` — returns canned JSON, writes `llm_calls`.
  - `_FakeBroker` — records submitted orders, returns canned fills.
  - `_FakeReconciler` — owns no state; just exposes `start/stop`.

This keeps the load-bearing tests (AC-9..AC-12) realistic without
touching live network.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from analyzer.anthropic_client import AnthropicClient
from analyzer.opus import OpusReviewer
from analyzer.sonnet import SonnetAnalyzer
from broker.protocol import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)
from execution.orders import OrderSubmitter
from execution.sizing import SizingEngine
from execution.validator import ProposalValidator
from journal.migrate import apply_migrations
from journal.models import (
    FilingRow,
    LlmCallRow,
    UniverseMemberRow,
    UniverseSnapshotRow,
)
from journal.repo import (
    JournalRepo,
    insert_filing,
    insert_llm_call,
    insert_universe_member,
    insert_universe_snapshot,
)
from killswitch.api import KillSwitch
from prefilter.orchestrator import Prefilter
from prompts.loader import PromptBuilder
from proposal.store import ProposalStore
from universe.api import Universe


@dataclass
class _FakeBroker:
    """Records `submit_order` invocations; returns canned account / quotes.

    Replaces `AlpacaBroker` at the BrokerAdapter Protocol boundary so the
    real execution / sizing / order-submission code paths run unchanged.
    """

    submitted_orders: list[OrderRequest] = field(default_factory=list)
    submitted_by_client_id: dict[str, SubmittedOrder] = field(default_factory=dict)
    # Pre-seeded orders simulating the broker's memory of a previous
    # process run whose journal write was lost mid-pipeline (M-4 test).
    preseeded_by_client_id: dict[str, SubmittedOrder] = field(default_factory=dict)
    # Open positions visible to consumers via `get_positions()`. Defaults
    # empty (no holdings); tests that exercise Feature B's capacity gate
    # can override this list to simulate a full KS-5 slate.
    positions: list[Position] = field(default_factory=list)
    equity: float = 100_000.0
    cash: float = 100_000.0
    quote_last: float = 100.0
    next_order_id: int = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self.submitted_orders.append(req)
        self.next_order_id += 1
        resp = SubmittedOrder(
            broker_order_id=f"fake-{self.next_order_id:06d}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            status="accepted",
            submitted_at=dt.datetime.now(dt.UTC),
            limit_price=req.limit_price,
            stop_price=req.stop_price,
        )
        self.submitted_by_client_id[req.client_order_id] = resp
        return resp

    def cancel_order(self, broker_order_id: str) -> None:
        return None

    def get_order_by_client_id(
        self, client_order_id: str
    ) -> SubmittedOrder | None:
        if client_order_id in self.submitted_by_client_id:
            return self.submitted_by_client_id[client_order_id]
        return self.preseeded_by_client_id.get(client_order_id)

    def replace_stop_order(
        self,
        broker_order_id: str,
        *,
        new_stop_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        # Atomic replace stub mirroring Alpaca's PATCH semantics: the
        # broker emits a new SubmittedOrder with the same protective
        # role but updated stop_price.
        self.next_order_id += 1
        resp = SubmittedOrder(
            broker_order_id=f"fake-replace-{self.next_order_id:06d}",
            client_order_id=client_order_id,
            symbol="REPLACED",
            side="sell",
            qty=1,
            order_type="stop",
            status="accepted",
            submitted_at=dt.datetime.now(dt.UTC),
            stop_price=new_stop_price,
        )
        self.submitted_by_client_id[client_order_id] = resp
        return resp

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self.equity,
            cash=self.cash,
            buying_power=self.cash,
            long_market_value=0.0,
            daypl=0.0,
            snapshot_at=dt.datetime.now(dt.UTC),
        )

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=self.quote_last - 0.01,
            ask=self.quote_last + 0.01,
            last=self.quote_last,
            ts=dt.datetime.now(dt.UTC),
        )


@dataclass
class _FakeAnthropicClient:
    """Stand-in for `AnthropicClient.call`. Mirrors the real contract:
    writes one `llm_calls` row per call so analyzer / reviewer can read
    telemetry back from the journal.

    `responses` is a queue keyed by `purpose` so the test fixture can
    pre-load distinct strings for analyze vs. review without ordering
    coupling.
    """

    db_path: str
    responses_by_purpose: dict[str, list[str]] = field(default_factory=dict)
    delay_seconds_by_decision_id: dict[str, float] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def call(
        self,
        request: Any,
        *,
        model_id: str,
        purpose: str,
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str:
        delay = self.delay_seconds_by_decision_id.get(decision_id, 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        self.calls.append(
            {
                "request": request,
                "model_id": model_id,
                "purpose": purpose,
                "decision_id": decision_id,
            }
        )
        queue = self.responses_by_purpose.get(purpose, [])
        if not queue:
            raise AssertionError(
                f"no fake response queued for purpose={purpose!r}"
            )
        text = queue.pop(0)
        insert_llm_call(
            self.db_path,
            LlmCallRow(
                decision_id=decision_id,
                purpose=purpose,
                model_id=model_id,
                prompt_version=request.prompt_version,
                input_tokens=1500,
                output_tokens=800,
                cache_read_tokens=4000,
                cache_creation_tokens=200,
                latency_ms=int(delay * 1000) if delay > 0 else 1234,
                cost_usd=0.0234,
                error_class=None,
            ),
        )
        return text


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


@pytest.fixture
def journal(db_path: str) -> JournalRepo:
    return JournalRepo(db_path)


@pytest.fixture
def killswitch(journal: JournalRepo) -> KillSwitch:
    return KillSwitch(journal)


@pytest.fixture
def universe(db_path: str) -> Universe:
    """Universe pre-seeded with the CIKs the smoke fixture filings use."""
    snap_id = insert_universe_snapshot(
        db_path,
        UniverseSnapshotRow(
            snapshot_date=dt.date(2026, 4, 28),
            sec_tickers_hash="x" * 64,
            alpaca_assets_hash="y" * 64,
            yfinance_failures=0,
            member_count=2,
            is_degraded=0,
        ),
    )
    insert_universe_member(
        db_path,
        UniverseMemberRow(
            snapshot_id=snap_id,
            cik=320193,
            ticker="ACME",
            exchange="NASDAQ",
            market_cap=10_000_000_000.0,
            prev_close=100.0,
        ),
    )
    insert_universe_member(
        db_path,
        UniverseMemberRow(
            snapshot_id=snap_id,
            cik=789019,
            ticker="WIDG",
            exchange="NASDAQ",
            market_cap=10_000_000_000.0,
            prev_close=100.0,
        ),
    )
    return Universe.load_latest(db_path)


@pytest.fixture
def prompts_dir() -> Path:
    return Path("src/prompts")


@pytest.fixture
def prompt_builder(prompts_dir: Path) -> PromptBuilder:
    return PromptBuilder(prompts_dir)


@pytest.fixture
def proposal_store(db_path: str) -> ProposalStore:
    return ProposalStore(db_path=db_path)


@pytest.fixture
def fake_broker() -> _FakeBroker:
    return _FakeBroker()


@pytest.fixture
def fake_anthropic(db_path: str) -> _FakeAnthropicClient:
    return _FakeAnthropicClient(db_path=db_path)


def _valid_proposal_json(*, conviction: int = 9, symbol: str = "ACME") -> str:
    payload = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": 9.50,
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
    return json.dumps(payload)


def _no_trade_json() -> str:
    payload = {
        "decision": "no_trade",
        "thesis_or_reason": (
            "8-K Item 8.01 contains routine corporate-governance disclosure with "
            "no material catalyst; conviction below threshold."
        ),
        "signals_considered": ["Item 8.01 — routine governance"],
    }
    return json.dumps(payload)


def _opus_ratify_json() -> str:
    payload = {
        "decision": "ratify",
        "rationale": (
            "Filing language supports the thesis with a concrete catalyst and "
            "plausible integration timeline. The proposal sizing is conservative."
        ),
    }
    return json.dumps(payload)


def _make_filing_row(
    db_path: str,
    *,
    accession: str,
    cik: int = 320193,
    form_type: str = "8-K",
    item_codes: str = '["1.01"]',
    issuer_ticker: str = "ACME",
    raw_text_dir: Path,
) -> FilingRow:
    """Insert a filing row + its raw text artifact; return the persisted row."""
    raw_path = raw_text_dir / f"{accession}.txt"
    raw_path.write_text(
        f"# Filing {accession}\nIssuer: {issuer_ticker}\n"
        "Material acquisition announced; integration timeline plausible.\n"
    )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number=accession,
            cik=cik,
            form_type=form_type,
            filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
            raw_text_path=str(raw_path),
            content_hash="hash-" + accession,
            item_codes=item_codes,
            issuer_ticker=issuer_ticker,
        ),
    )
    from journal.repo import get_filing_by_id
    f = get_filing_by_id(db_path, fid)
    assert f is not None
    return f


@pytest.fixture
def make_filing(db_path: str, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    def _factory(**kwargs: Any) -> FilingRow:
        return _make_filing_row(db_path, raw_text_dir=raw_dir, **kwargs)
    return _factory


def _build_components(
    *,
    db_path: str,
    journal: JournalRepo,
    universe: Universe,
    prompt_builder: PromptBuilder,
    proposal_store: ProposalStore,
    fake_broker: _FakeBroker,
    fake_anthropic: _FakeAnthropicClient,
    killswitch: KillSwitch,
    queue: asyncio.Queue,
    similarity_artifact_dir: str,
    ks5_max_concurrent: int | None = None,
) -> Any:
    """Build the AgentComponents bundle used by all integration tests.

    The Prefilter uses a `MagicMock` similarity checker because we
    pre-fill `prefilter_decisions` rows or rely on the form-4 / 8-K
    bypass paths — the smoke filings are 8-Ks with allow-list item
    codes (material_8k_bypass), which the orchestrator routes around
    similarity entirely.
    """
    from app.composition import AgentComponents

    similarity = MagicMock()
    prefilter = Prefilter(db_path=db_path, universe=universe, similarity=similarity)
    sonnet_model_id = "claude-sonnet-4-6"
    opus_model_id = "claude-opus-4-7"

    # Capped Anthropic clients per analyzer / reviewer (the real client
    # is one instance; here we use the same fake for both because it's
    # purpose-keyed).
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id=opus_model_id,
        ks4_pct_cap=0.20,
        db_path=db_path,
    )
    analyzer = SonnetAnalyzer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_reviewer=reviewer,
        sonnet_model_id=sonnet_model_id,
        opus_review_conviction_threshold=7,
        db_path=db_path,
        ks5_max_concurrent=ks5_max_concurrent,
        open_positions_counter=(
            (lambda: len(fake_broker.get_positions()))
            if ks5_max_concurrent is not None
            else None
        ),
    )
    validator = ProposalValidator()
    sizer = SizingEngine()
    submitter = OrderSubmitter()

    # Reconciler is non-essential for the smoke tests; we wire a stub
    # that has start/stop methods so the loop's lifecycle works.
    reconciler = _StubReconciler()

    # Register prompt versions so FK constraints hold (S5.2 D-1 carry-fwd).
    from app.composition import register_composed_prompt_versions
    register_composed_prompt_versions(prompt_builder, db_path)

    return AgentComponents(
        universe=universe,
        prefilter=prefilter,
        analyzer=analyzer,
        reviewer=reviewer,
        proposal_store=proposal_store,
        validator=validator,
        sizer=sizer,
        submitter=submitter,
        execution=None,  # legacy alias unused; story spec lists 'execution' but the
                        # validator + sizer + submitter trio is the actual surface
        broker=fake_broker,
        reconciler=reconciler,
        killswitch=killswitch,
        ingestion_queue=queue,
        journal=journal,
    )


class _StubReconciler:
    """Reconciler stand-in for tests; implements start/stop only."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def build_components(
    db_path: str,
    journal: JournalRepo,
    universe: Universe,
    prompt_builder: PromptBuilder,
    proposal_store: ProposalStore,
    fake_broker: _FakeBroker,
    fake_anthropic: _FakeAnthropicClient,
    killswitch: KillSwitch,
    tmp_path: Path,
):
    def _factory(
        *,
        queue: asyncio.Queue,
        ks5_max_concurrent: int | None = None,
    ) -> Any:
        return _build_components(
            db_path=db_path,
            journal=journal,
            universe=universe,
            prompt_builder=prompt_builder,
            proposal_store=proposal_store,
            fake_broker=fake_broker,
            fake_anthropic=fake_anthropic,
            killswitch=killswitch,
            queue=queue,
            similarity_artifact_dir=str(tmp_path / "sim"),
            ks5_max_concurrent=ks5_max_concurrent,
        )
    return _factory


# Re-export helpers so test modules can import them directly.
__all__ = [
    "_FakeAnthropicClient",
    "_FakeBroker",
    "_StubReconciler",
    "_make_filing_row",
    "_no_trade_json",
    "_opus_ratify_json",
    "_valid_proposal_json",
    "build_components",
]
