"""S9.1 — proposals page (AC-4).

`/proposals` lists recent proposals; each row is collapsible to show full
Sonnet `raw_response` + Opus review (if any). Filterable by `?symbol=` and
`?status=`. `?limit=` accepted up to 1000.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from journal.models import ProposalRow
from journal.repo import insert_proposal

from .conftest import (
    auth_headers,
    seed_filings,
    seed_prompt,
    seed_proposals,
    seed_review,
)


def test_proposals_lists_seeded_proposals(
    client: TestClient, journal, db_path: str
) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=3)
    pids = seed_proposals(journal, filing_ids=fids)
    resp = client.get("/proposals", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    for i in range(len(pids)):
        assert f"AB{i}" in body  # symbol column rendered
        assert f"d-{i:03d}" in body  # decision_id rendered


def test_proposals_includes_raw_response_collapsible(
    client: TestClient, journal, db_path: str
) -> None:
    """AC-4: raw_response visible in collapsible <details> (HTML element,
    no JS required).
    """
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    seed_proposals(journal, filing_ids=fids)
    resp = client.get("/proposals", headers=auth_headers())
    body = resp.text
    assert "<details" in body
    # raw_response content reaches the page (even if HTML-escaped — the
    # security test asserts escaping; here we assert it is _rendered_).
    assert "alpha thesis" in body or "thesis" in body


def test_proposals_renders_opus_review_when_present(
    client: TestClient, journal, db_path: str
) -> None:
    seed_prompt(db_path)
    seed_prompt(db_path, prompt_version="opus:v1#xyz")
    fids = seed_filings(db_path, n=1)
    pids = seed_proposals(journal, filing_ids=fids)
    seed_review(journal, pids[0], "opus:v1#xyz")
    resp = client.get("/proposals", headers=auth_headers())
    body = resp.text
    assert "approved" in body.lower()
    # Opus model_id flagged so operator sees both calls
    assert "opus" in body.lower()


def test_proposals_filter_by_symbol(client: TestClient, journal, db_path: str) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=3)
    seed_proposals(journal, filing_ids=fids)
    resp = client.get("/proposals?symbol=AB1", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    assert "AB1" in body
    # AB0 + AB2 must be filtered out.
    assert "d-000" not in body
    assert "d-002" not in body


def test_proposals_limit_param_caps_at_1000(client: TestClient) -> None:
    """AC-4: `?limit=` query param accepted up to 1000; overflow rejected as 400."""
    resp = client.get("/proposals?limit=2000", headers=auth_headers())
    assert resp.status_code == 400


def test_proposals_negative_limit_rejected(client: TestClient) -> None:
    resp = client.get("/proposals?limit=-1", headers=auth_headers())
    assert resp.status_code == 400


def test_proposals_no_trade_row_renders_kind_and_thesis_or_reason(
    client: TestClient, journal, db_path: str
) -> None:
    """P4: no-trade rows store NULL for symbol/direction/conviction/thesis;
    the operator-visible reason lives in `raw_response.thesis_or_reason`.
    The dashboard must surface `kind` and fall back to the parsed reason
    so the row isn't blank.
    """
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    reason = (
        "Filing is a routine S-1/A registration amendment with no new "
        "operating data, no guidance, and no insider transactions. Pass."
    )
    raw = json.dumps(
        {
            "decision": "no_trade",
            "thesis_or_reason": reason,
            "signals_considered": ["form_type", "no_financials"],
        }
    )
    insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fids[0],
            decision_id="d-no-trade-1",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet:v1#abc",
            raw_response=raw,
            kind="no_trade",
            symbol=None,
            direction=None,
            size_pct_requested=None,
            conviction=None,
            thesis=None,
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=400,
            cache_creation_tokens=600,
            latency_ms=2500,
            cost_usd=0.012,
        ),
    )
    resp = client.get("/proposals", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    assert "no_trade" in body
    assert "routine S-1/A registration" in body
    assert "thesis / reason" in body


def test_proposals_no_trade_row_with_invalid_raw_response_does_not_500(
    client: TestClient, journal, db_path: str
) -> None:
    """If raw_response is malformed JSON, we render an empty thesis cell —
    not a 500. (Defensive — the page is read-only operator UI.)"""
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fids[0],
            decision_id="d-no-trade-broken",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet:v1#abc",
            raw_response="<<<not json>>>",
            kind="no_trade",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    resp = client.get("/proposals", headers=auth_headers())
    assert resp.status_code == 200
    assert "no_trade" in resp.text
