"""S5.5 — Proposal store tests.

Architecture references: §2.4 (Proposal store), §3.2 (TradeProposal schema),
§3.3 (NoTradeRecord schema), FR-19, FR-27, FR-28, FR-29, FR-30, NFR-16.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from analyzer.telemetry import CallTelemetry
from journal.migrate import apply_migrations
from journal.models import FilingRow, PromptRow
from journal.repo import (
    get_proposal_by_decision_id,
    get_proposal_by_id,
    get_proposal_review_by_proposal_id,
    insert_filing,
    insert_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


@pytest.fixture
def prompt_version(db: str) -> str:
    pv = "sonnet_filing_analysis_v1@feedf00d0001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="feedf00d0001" + "0" * 52,
        ),
    )
    return pv


@pytest.fixture
def filing(db: str) -> FilingRow:
    """Insert a filing and return the row with `id` populated."""
    fid = insert_filing(
        db,
        FilingRow(
            accession_number="0001234567-26-000123",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
            raw_text_path="/var/lib/quinn/raw/0001234567-26-000123.txt",
            content_hash="aaa111",
            item_codes='["1.01", "8.01"]',
            issuer_ticker="ACME",
        ),
    )
    from journal.repo import get_filing_by_id
    f = get_filing_by_id(db, fid)
    assert f is not None
    return f


def _telemetry() -> CallTelemetry:
    return CallTelemetry(
        input_tokens=1500,
        output_tokens=800,
        cache_read_tokens=4000,
        cache_creation_tokens=200,
        latency_ms=4321,
        cost_usd=0.0234,
    )


def _valid_trade_proposal_payload() -> dict:
    """A minimum-valid TradeProposal payload per architecture §3.2."""
    return {
        "symbol": "ACME",
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": 9.50,
        "time_horizon_days": 14,
        "conviction": 8,
        "thesis": (
            "ACME announced a material acquisition in 8-K Item 1.01 "
            "with concrete pricing terms; integration timeline is plausible."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 trading days"],
        "risk_factors": ["Closing conditions not yet met"],
    }


def _valid_no_trade_payload() -> dict:
    """A minimum-valid NoTradeRecord payload per architecture §3.3."""
    return {
        "decision": "no_trade",
        "thesis_or_reason": (
            "8-K Item 8.01 contains routine corporate-governance disclosure with "
            "no material catalyst; conviction below threshold."
        ),
        "signals_considered": ["Item 8.01 — routine governance"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_store_valid_proposal_persists_with_system_fields(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-1, AC-3, AC-5: store a valid TradeProposal; the persisted
    `raw_response` carries the enriched JSON (LLM payload + system-injected
    `prompt_version` and `source_filings`); telemetry maps to row columns."""
    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    payload = _valid_trade_proposal_payload()
    raw = json.dumps(payload)
    result = ProposalEmitted(payload=payload, raw_response=raw)

    pid = store.store(
        result,
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-store-001",
        raw_response=raw,
        telemetry=_telemetry(),
    )
    assert pid > 0

    row = get_proposal_by_id(db, pid)
    assert row is not None
    assert row.kind == "trade_proposal"
    assert row.decision_id == "dec-store-001"
    assert row.symbol == "ACME"
    assert row.direction == "long"
    assert row.conviction == 8
    assert row.size_pct_requested == pytest.approx(0.10)
    assert row.input_tokens == 1500
    assert row.output_tokens == 800
    assert row.cache_read_tokens == 4000
    assert row.cache_creation_tokens == 200
    assert row.latency_ms == 4321
    assert row.cost_usd == pytest.approx(0.0234)

    # AC-3: raw_response is the *enriched* JSON — system fields injected.
    enriched = json.loads(row.raw_response)
    assert enriched["prompt_version"] == prompt_version
    assert enriched["source_filings"] == [
        {
            "accession_number": filing.accession_number,
            "filing_type": filing.form_type,
            "item_codes": ["1.01", "8.01"],
        }
    ]
    # Original LLM-emitted fields preserved verbatim.
    assert enriched["symbol"] == "ACME"
    assert enriched["thesis"].startswith("ACME announced a material acquisition")


def test_store_no_trade_persists(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-1, AC-2: store a NoTradeRecord with kind='no_trade'; symbol /
    direction / conviction / size are NULL on the row."""
    from analyzer.results import NoTrade
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    payload = _valid_no_trade_payload()
    raw = json.dumps(payload)
    pid = store.store(
        NoTrade(payload=payload, raw_response=raw),
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-store-002",
        raw_response=raw,
        telemetry=_telemetry(),
    )
    row = get_proposal_by_id(db, pid)
    assert row is not None
    assert row.kind == "no_trade"
    assert row.symbol is None
    assert row.direction is None
    assert row.conviction is None
    assert row.size_pct_requested is None
    # Telemetry still populated.
    assert row.input_tokens == 1500
    assert row.cost_usd == pytest.approx(0.0234)


def test_invalid_proposal_raises_schema_error(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-2: schema-invalid payload → ProposalSchemaError, no row written."""
    from analyzer.results import ProposalEmitted
    from proposal.schemas import ProposalSchemaError
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    bad_payload = _valid_trade_proposal_payload()
    bad_payload["size_pct_of_capital"] = 0.99  # > 0.20 cap
    raw = json.dumps(bad_payload)

    with pytest.raises(ProposalSchemaError):
        store.store(
            ProposalEmitted(payload=bad_payload, raw_response=raw),
            filing=filing,
            model_id="claude-sonnet-4-6",
            prompt_version=prompt_version,
            decision_id="dec-store-bad-001",
            raw_response=raw,
            telemetry=_telemetry(),
        )
    # No row written.
    assert get_proposal_by_decision_id(db, "dec-store-bad-001") is None


def test_idempotent_on_decision_id(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-4: re-storing the same decision_id returns the same proposal_id
    without writing a new row."""
    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    payload = _valid_trade_proposal_payload()
    raw = json.dumps(payload)
    args = dict(
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-store-003",
        raw_response=raw,
        telemetry=_telemetry(),
    )
    pid1 = store.store(ProposalEmitted(payload=payload, raw_response=raw), **args)
    pid2 = store.store(ProposalEmitted(payload=payload, raw_response=raw), **args)
    assert pid1 == pid2
    # Sanity: only one row in the table for that decision_id.
    row = get_proposal_by_decision_id(db, "dec-store-003")
    assert row is not None
    assert row.id == pid1


def test_review_persisted_uniquely_per_proposal(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-6: store_review writes to proposal_reviews; UNIQUE(proposal_id)
    is enforced — second store_review on the same proposal_id raises."""
    import sqlite3

    from analyzer.opus import OpusRatified
    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    payload = _valid_trade_proposal_payload()
    raw = json.dumps(payload)
    pid = store.store(
        ProposalEmitted(payload=payload, raw_response=raw),
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-store-004",
        raw_response=raw,
        telemetry=_telemetry(),
    )

    proposal_row = get_proposal_by_id(db, pid)
    assert proposal_row is not None
    review = OpusRatified(
        proposal=proposal_row,
        rationale="Filing language supports the thesis with concrete catalyst.",
    )
    review_raw = json.dumps({"decision": "ratify", "rationale": review.rationale})
    store.store_review(
        pid,
        review,
        model_id="claude-opus-4-7",
        prompt_version=prompt_version,
        raw_response=review_raw,
        telemetry=_telemetry(),
    )
    row = get_proposal_review_by_proposal_id(db, pid)
    assert row is not None
    assert row.decision == "ratify"
    assert row.rationale == review.rationale

    with pytest.raises(sqlite3.IntegrityError):
        store.store_review(
            pid,
            review,
            model_id="claude-opus-4-7",
            prompt_version=prompt_version,
            raw_response=review_raw,
            telemetry=_telemetry(),
        )


def test_telemetry_fields_round_trip(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-5: every CallTelemetry field round-trips into the proposal row."""
    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    tel = CallTelemetry(
        input_tokens=12345,
        output_tokens=678,
        cache_read_tokens=9012,
        cache_creation_tokens=345,
        latency_ms=99999,
        cost_usd=0.5678,
    )
    store = ProposalStore(db_path=db)
    payload = _valid_trade_proposal_payload()
    raw = json.dumps(payload)
    pid = store.store(
        ProposalEmitted(payload=payload, raw_response=raw),
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-tel-001",
        raw_response=raw,
        telemetry=tel,
    )
    row = get_proposal_by_id(db, pid)
    assert row is not None
    assert row.input_tokens == tel.input_tokens
    assert row.output_tokens == tel.output_tokens
    assert row.cache_read_tokens == tel.cache_read_tokens
    assert row.cache_creation_tokens == tel.cache_creation_tokens
    assert row.latency_ms == tel.latency_ms
    assert row.cost_usd == pytest.approx(tel.cost_usd)


def test_replay_data_sufficient(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """AC-7: persisted proposal carries decision_id, prompt_version,
    raw_response, and source_filings sufficient to reconstruct the API
    request (ADR-001 replay)."""
    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    payload = _valid_trade_proposal_payload()
    raw = json.dumps(payload)
    pid = store.store(
        ProposalEmitted(payload=payload, raw_response=raw),
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-replay-001",
        raw_response=raw,
        telemetry=_telemetry(),
    )

    row = store.fetch(pid)
    enriched = json.loads(row.raw_response)
    # Replay builds: prompt_version → which loader composition to use;
    # source_filings → the FilingRow to re-fetch raw text from;
    # raw_response → expected output for diff.
    assert row.decision_id == "dec-replay-001"
    assert row.prompt_version == prompt_version
    assert enriched["prompt_version"] == prompt_version
    assert len(enriched["source_filings"]) == 1
    src = enriched["source_filings"][0]
    assert src["accession_number"] == filing.accession_number
    assert src["filing_type"] == filing.form_type
    assert src["item_codes"] == ["1.01", "8.01"]


def test_malformed_persists_with_reasoning_notes(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """S5.3 AC-6 carry-forward: AnalyzerMalformed gets a row (kind=no_trade)
    with reasoning_notes capturing the parse failure."""
    from analyzer.results import AnalyzerMalformed
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    raw = "this is not valid JSON {[}"
    pid = store.store(
        AnalyzerMalformed(raw_response=raw, error="Expecting value: line 1 col 1"),
        filing=filing,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        decision_id="dec-malformed-001",
        raw_response=raw,
        telemetry=_telemetry(),
    )
    row = get_proposal_by_id(db, pid)
    assert row is not None
    assert row.kind == "no_trade"
    assert row.reasoning_notes is not None
    assert "analyzer_malformed" in row.reasoning_notes
    # raw_response preserved verbatim — the LLM didn't emit a parseable
    # object so there's nowhere to inject system fields.
    assert row.raw_response == raw


def test_store_review_handles_all_opus_paths(
    db: str, filing: FilingRow, prompt_version: str
) -> None:
    """FR-28: every Opus path (ratify, modify, reject, malformed) produces
    a `proposal_reviews` row. modifications_json populated only on modify."""
    from analyzer.opus import OpusMalformed, OpusModified, OpusRatified, OpusRejected
    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)

    paths = [
        ("ratify", OpusRatified, {}, None),
        (
            "modify",
            OpusModified,
            {"size_pct_of_capital": 0.05, "stop_loss_price": 9.00},
            None,
        ),
        ("reject", OpusRejected, {}, None),
        ("malformed", OpusMalformed, {}, "JSON parse failure"),
    ]
    for i, (label, cls, mods, error) in enumerate(paths):
        # Fresh proposal per path — proposal_reviews is UNIQUE(proposal_id).
        payload = _valid_trade_proposal_payload()
        raw = json.dumps(payload)
        pid = store.store(
            ProposalEmitted(payload=payload, raw_response=raw),
            filing=filing,
            model_id="claude-sonnet-4-6",
            prompt_version=prompt_version,
            decision_id=f"dec-paths-{i:03d}",
            raw_response=raw,
            telemetry=_telemetry(),
        )
        proposal_row = get_proposal_by_id(db, pid)
        assert proposal_row is not None

        if cls is OpusRatified:
            review = OpusRatified(
                proposal=proposal_row,
                rationale=("Concrete catalyst supports the thesis. " * 2).strip(),
            )
        elif cls is OpusModified:
            working = proposal_row.model_copy(
                update={"size_pct_requested": mods["size_pct_of_capital"]}
            )
            review = OpusModified(
                proposal=proposal_row,
                working_proposal=working,
                rationale=("Tighten size for borderline conviction. " * 2).strip(),
                modifications=mods,
            )
        elif cls is OpusRejected:
            review = OpusRejected(
                proposal=proposal_row,
                rationale=("Filing language non-material. " * 3).strip(),
            )
        else:  # OpusMalformed
            review = OpusMalformed(
                proposal=proposal_row, raw_response="bad json", error=error or "x"
            )

        review_raw = json.dumps({"decision": label, "rationale": "..."})
        store.store_review(
            pid,
            review,
            model_id="claude-opus-4-7",
            prompt_version=prompt_version,
            raw_response=review_raw,
            telemetry=_telemetry(),
        )
        row = get_proposal_review_by_proposal_id(db, pid)
        assert row is not None
        assert row.decision == label
        if cls is OpusModified:
            assert row.modifications_json is not None
            assert json.loads(row.modifications_json) == mods
        else:
            assert row.modifications_json is None


def test_unregistered_prompt_version_raises_after_api_call(
    db: str, filing: FilingRow
) -> None:
    """Carry-forward from S5.2 review (T-1, medium): if the composed
    prompt_version was NOT pre-registered in the `prompts` table, the
    `proposals.prompt_version` FK fires at insert time, AFTER any upstream
    API call would have charged. This test pins the failure mode so S5.6's
    wiring (which is responsible for `insert_prompt` BEFORE `call`) can be
    validated against the same hazard.

    Per D-031: composed prompt_version is `{name}@{sha256(composed)[:12]}`
    and is the FK target on `proposals.prompt_version`. Forgetting the
    registration is a real production-incident shape — money spent, no row.
    """
    import sqlite3

    from analyzer.results import ProposalEmitted
    from proposal.store import ProposalStore

    store = ProposalStore(db_path=db)
    payload = _valid_trade_proposal_payload()
    raw = json.dumps(payload)
    unregistered_pv = "sonnet_filing_analysis_v1@notregistered"

    with pytest.raises(sqlite3.IntegrityError):
        store.store(
            ProposalEmitted(payload=payload, raw_response=raw),
            filing=filing,
            model_id="claude-sonnet-4-6",
            prompt_version=unregistered_pv,
            decision_id="dec-fk-001",
            raw_response=raw,
            telemetry=_telemetry(),
        )
    # Confirm the row was NOT inserted (FK violation rolls back).
    assert get_proposal_by_decision_id(db, "dec-fk-001") is None
