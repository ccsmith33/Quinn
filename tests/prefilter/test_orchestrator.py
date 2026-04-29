"""S4.3 — Prefilter orchestrator tests.

Composes universe gate + 8-K item codes (S4.1) + similarity (S4.2) per the
order in story-04-03 AC-2; persists every decision to `prefilter_decisions`
(FR-14); idempotent on `filing_id`.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    FilingRow,
    UniverseMemberRow,
    UniverseSnapshotRow,
)
from journal.repo import (
    get_prefilter_decision_by_filing,
    insert_filing,
    insert_universe_member,
    insert_universe_snapshot,
)
from prefilter.orchestrator import Prefilter, PrefilterDecision
from prefilter.similarity import SimilarityChecker
from universe.api import Universe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    d = tmp_path / "similarity"
    d.mkdir()
    return d


def _seed_universe(db_path: str, *, ciks: list[int]) -> Universe:
    """Create a single-snapshot universe containing the given CIKs."""
    snap_id = insert_universe_snapshot(
        db_path,
        UniverseSnapshotRow(
            snapshot_date=dt.date(2026, 4, 28),
            sec_tickers_hash="x" * 64,
            alpaca_assets_hash="y" * 64,
            yfinance_failures=0,
            member_count=len(ciks),
            is_degraded=0,
        ),
    )
    for i, cik in enumerate(ciks):
        insert_universe_member(
            db_path,
            UniverseMemberRow(
                snapshot_id=snap_id,
                cik=cik,
                ticker=f"TKR{i:03d}",
                exchange="NASDAQ",
                market_cap=10_000_000_000.0,
                prev_close=100.0,
            ),
        )
    return Universe.load_latest(db_path)


@pytest.fixture
def universe(db_path: str) -> Universe:
    return _seed_universe(db_path, ciks=[320193, 789019])


@pytest.fixture
def similarity(db_path: str, artifact_dir: Path) -> SimilarityChecker:
    return SimilarityChecker(
        db_path=db_path, artifact_dir=str(artifact_dir), threshold=0.97
    )


@pytest.fixture
def prefilter(
    db_path: str,
    universe: Universe,
    similarity: SimilarityChecker,
) -> Prefilter:
    return Prefilter(
        db_path=db_path, universe=universe, similarity=similarity
    )


def _filing(
    *,
    db_path: str,
    accession: str,
    cik: int,
    form_type: str,
    item_codes: list[str] | None = None,
    filed_at: dt.datetime | None = None,
) -> FilingRow:
    if filed_at is None:
        filed_at = dt.datetime(2026, 4, 1, 9, 0, 0)
    row = FilingRow(
        accession_number=accession,
        cik=cik,
        form_type=form_type,
        filed_at=filed_at,
        fetched_at=filed_at + dt.timedelta(seconds=30),
        raw_text_path="/tmp/unused.txt",
        content_hash="0" * 64,
        item_codes=json.dumps(item_codes) if item_codes is not None else None,
    )
    new_id = insert_filing(db_path, row)
    return row.model_copy(update={"id": new_id})


# ---------------------------------------------------------------------------
# AC-2.1 — universe gate
# ---------------------------------------------------------------------------


def test_out_of_universe_rejected_at_orchestrator(
    prefilter: Prefilter, db_path: str
) -> None:
    f = _filing(
        db_path=db_path,
        accession="0000999-26-000001",
        cik=999999,  # not in seeded universe
        form_type="10-K",
    )
    decision = prefilter.evaluate(f, "any text")

    assert isinstance(decision, PrefilterDecision)
    assert decision.decision == "reject"
    assert decision.rule_fired == "universe"

    persisted = get_prefilter_decision_by_filing(db_path, f.id or 0)
    assert persisted is not None
    assert persisted.decision == "reject"
    assert persisted.rule_fired == "universe"


# ---------------------------------------------------------------------------
# AC-2.2 — 8-K deny short-circuits before similarity
# ---------------------------------------------------------------------------


def test_8k_deny_only_rejected_before_similarity(
    db_path: str, universe: Universe, artifact_dir: Path
) -> None:
    sim_spy = MagicMock(spec=SimilarityChecker)
    pref = Prefilter(db_path=db_path, universe=universe, similarity=sim_spy)

    f = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="8-K",
        item_codes=["5.04", "5.05"],  # all deny
    )
    decision = pref.evaluate(f, "boilerplate text")

    assert decision.decision == "reject"
    assert decision.rule_fired == "item_code_deny"
    sim_spy.check.assert_not_called()


# ---------------------------------------------------------------------------
# AC-2.3 — material 8-K bypass: allow-list 8-K skips similarity
# ---------------------------------------------------------------------------


def test_material_8k_bypasses_similarity(
    db_path: str, universe: Universe, similarity: SimilarityChecker
) -> None:
    """An 8-K with at least one allow-list item code is accepted without
    invoking similarity, even if a near-identical prior 8-K exists."""
    sim_spy = MagicMock(wraps=similarity)
    pref = Prefilter(db_path=db_path, universe=universe, similarity=sim_spy)

    # Seed a "prior" 8-K via the real similarity checker so cache is populated.
    prior_text = "Earnings release. Net income was $1B. Revenue $30B." * 50
    f_prior = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="8-K",
        item_codes=["2.02"],
        filed_at=dt.datetime(2026, 1, 30, 9, 0, 0),
    )
    pref.evaluate(f_prior, prior_text)
    sim_spy.check.reset_mock()

    f_new = _filing(
        db_path=db_path,
        accession="0000001-26-000002",
        cik=320193,
        form_type="8-K",
        item_codes=["2.02"],  # allow list
        filed_at=dt.datetime(2026, 4, 30, 9, 0, 0),
    )
    decision = pref.evaluate(f_new, prior_text)  # identical text

    assert decision.decision == "accept"
    assert decision.rule_fired == "material_8k_bypass"
    sim_spy.check.assert_not_called()


# ---------------------------------------------------------------------------
# AC-2.4 — Form 4 skips similarity
# ---------------------------------------------------------------------------


def test_form_4_accepted_no_similarity_call(
    db_path: str, universe: Universe
) -> None:
    sim_spy = MagicMock(spec=SimilarityChecker)
    pref = Prefilter(db_path=db_path, universe=universe, similarity=sim_spy)

    f = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="4",
    )
    decision = pref.evaluate(f, "any text")

    assert decision.decision == "accept"
    assert decision.rule_fired == "form_4_universe_only"
    sim_spy.check.assert_not_called()


# ---------------------------------------------------------------------------
# AC-2.5 — 10-Q near-duplicate rejected by similarity
# ---------------------------------------------------------------------------


def _ten_q_template(quarter_label: str, eps: float, revenue: float) -> str:
    boilerplate = (
        "Management's discussion and analysis of financial condition and results "
        "of operations. The discussion contains forward looking statements that "
        "involve risks and uncertainties. We design develop and sell consumer "
        "electronics. Our products are sold through our retail and online stores "
        "and through cellular network carriers wholesalers retailers and "
        "resellers. Critical accounting policies and estimates the preparation "
        "of consolidated financial statements requires management to make "
        "estimates and assumptions. Liquidity and capital resources the Company "
        "believes that its balances of cash and marketable securities along with "
        "cash generated by ongoing operations will be sufficient to satisfy its "
        "cash requirements and capital return program. Recent accounting "
        "pronouncements the Company adopted the standard requiring lessees to "
        "recognize most leases on the balance sheet. " * 4
    )
    return (
        f"For the {quarter_label} quarter, diluted earnings per share were "
        f"${eps:.2f} and total net sales were ${revenue:.2f} billion. {boilerplate}"
    )


def test_10q_near_duplicate_rejected_by_similarity(
    prefilter: Prefilter, db_path: str
) -> None:
    f1 = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="10-Q",
        filed_at=dt.datetime(2026, 1, 30, 9, 0, 0),
    )
    assert prefilter.evaluate(f1, _ten_q_template("Q1", 1.50, 90.00)).decision == "accept"

    f2 = _filing(
        db_path=db_path,
        accession="0000001-26-000002",
        cik=320193,
        form_type="10-Q",
        filed_at=dt.datetime(2026, 4, 30, 9, 0, 0),
    )
    decision = prefilter.evaluate(f2, _ten_q_template("Q2", 1.55, 92.00))

    assert decision.decision == "reject"
    assert decision.rule_fired in {"similarity_minhash", "similarity_tfidf"}
    persisted = get_prefilter_decision_by_filing(db_path, f2.id or 0)
    assert persisted is not None
    assert persisted.similarity_score is not None
    assert persisted.similarity_score >= 0.97


# ---------------------------------------------------------------------------
# AC-3 — every decision persisted; AC-3 idempotent
# ---------------------------------------------------------------------------


def test_decision_persisted_to_journal(
    prefilter: Prefilter, db_path: str
) -> None:
    f = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="10-Q",
    )
    prefilter.evaluate(f, "some novel text " * 100)
    row = get_prefilter_decision_by_filing(db_path, f.id or 0)

    assert row is not None
    assert row.filing_id == f.id
    assert row.decision in {"accept", "reject"}
    assert row.rule_fired


def test_idempotent_evaluate(prefilter: Prefilter, db_path: str) -> None:
    """A second call with the same filing_id must not insert a duplicate row.

    `prefilter_decisions` is `UNIQUE(filing_id)`. The orchestrator must check
    for an existing decision and short-circuit on replay.
    """
    f = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="10-Q",
    )
    d1 = prefilter.evaluate(f, "some novel text " * 100)
    d2 = prefilter.evaluate(f, "completely different content " * 100)

    # Idempotent: the same decision is returned, and only one row is persisted.
    assert d1 == d2

    from journal.repo import connect

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS c FROM prefilter_decisions WHERE filing_id = ?",
            (f.id,),
        ).fetchone()
    assert rows["c"] == 1


# ---------------------------------------------------------------------------
# AC-4 — e2e p95 latency under 5s
# ---------------------------------------------------------------------------


def test_e2e_p95_latency_under_5s(
    db_path: str, artifact_dir: Path
) -> None:
    """Synthetic 200-filing day across 200 issuers → p95 < 5s end-to-end."""
    universe = _seed_universe(db_path, ciks=[900000 + i for i in range(200)])
    sim = SimilarityChecker(
        db_path=db_path, artifact_dir=str(artifact_dir), threshold=0.97
    )
    pref = Prefilter(db_path=db_path, universe=universe, similarity=sim)

    paragraph = (
        "Risk factors. Investing in the company involves a high degree of risk. "
        "Our business operations and financial results are subject to various "
        "risks and uncertainties including those described below which could "
        "adversely affect our business financial condition and results of "
        "operations. " * 200
    )
    base = dt.datetime(2026, 1, 1, 9, 0, 0)
    n = 200
    durations: list[float] = []
    for i in range(n):
        f = _filing(
            db_path=db_path,
            accession=f"0000{i:03d}-26-000001",
            cik=900000 + i,
            form_type="10-K",
            filed_at=base + dt.timedelta(minutes=i),
        )
        text = f"Filing number {i} {paragraph}"
        t0 = time.perf_counter()
        pref.evaluate(f, text)
        durations.append(time.perf_counter() - t0)

    durations.sort()
    p95 = durations[int(0.95 * n) - 1]
    assert p95 < 5.0, f"p95={p95:.3f}s exceeded 5s budget"


# ---------------------------------------------------------------------------
# AC-5 — additional ADR-003 e2e scenarios
# ---------------------------------------------------------------------------


def test_8k_item_2_02_above_similarity_threshold_accepted_via_bypass(
    prefilter: Prefilter, db_path: str
) -> None:
    """ADR-003 §"Material 8-K bypass" scenario: an earnings 8-K (item 2.02)
    whose text would otherwise be rejected by similarity is still accepted."""
    same_text = "Earnings press release. Q1 EPS $1.50 revenue $30B. " * 200

    f1 = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="8-K",
        item_codes=["2.02"],
        filed_at=dt.datetime(2026, 1, 30, 9, 0, 0),
    )
    assert prefilter.evaluate(f1, same_text).decision == "accept"

    f2 = _filing(
        db_path=db_path,
        accession="0000001-26-000002",
        cik=320193,
        form_type="8-K",
        item_codes=["2.02"],
        filed_at=dt.datetime(2026, 4, 30, 9, 0, 0),
    )
    decision = prefilter.evaluate(f2, same_text)
    assert decision.decision == "accept"
    assert decision.rule_fired == "material_8k_bypass"


def test_8k_with_no_item_codes_rejected_at_item_code_stage(
    prefilter: Prefilter, db_path: str
) -> None:
    """An 8-K filing with `item_codes=None` falls into the empty-list branch
    of the item-code prefilter (S4.1). The orchestrator must record that
    rejection without invoking similarity."""
    f = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="8-K",
        item_codes=None,
    )
    decision = prefilter.evaluate(f, "any text")

    assert decision.decision == "reject"
    assert decision.rule_fired == "item_code_empty"


def test_decision_returned_matches_decision_persisted(
    prefilter: Prefilter, db_path: str
) -> None:
    """Round-trip: caller's returned PrefilterDecision == persisted row's fields."""
    f = _filing(
        db_path=db_path,
        accession="0000001-26-000001",
        cik=320193,
        form_type="10-Q",
    )
    decision = prefilter.evaluate(f, "novel content " * 100)
    persisted: Any = get_prefilter_decision_by_filing(db_path, f.id or 0)
    assert persisted is not None
    assert persisted.decision == decision.decision
    assert persisted.rule_fired == decision.rule_fired
    assert persisted.similarity_score == decision.similarity_score
