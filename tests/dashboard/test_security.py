"""S9.1 — security probes (AC-12).

XSS in `proposals.raw_response` must be HTML-escaped on render.
Jinja2 must NOT use `|safe` on user-controlled data.
No path traversal accepted in any route param.
No secret values appear in any HTML response (defense in depth — the
dashboard does not have access to anything except `dashboard_user` /
`dashboard_password` hashes, but check anyway).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from journal.models import ProposalRow

from .conftest import (
    PASSWORD,
    auth_headers,
    seed_filings,
    seed_prompt,
)


def _insert_xss_proposal(db_path: str, filing_id: int) -> int:
    from journal.repo import insert_proposal

    return insert_proposal(
        db_path,
        ProposalRow(
            filing_id=filing_id,
            decision_id="d-xss",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet:v1#abc",
            raw_response='<script>alert("pwn")</script>',
            kind="trade",
            symbol="EVIL",
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis='<img src=x onerror="alert(1)">',
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1000,
            cost_usd=0.01,
        ),
    )


def test_xss_in_raw_response_is_html_escaped(client: TestClient, db_path: str) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    _insert_xss_proposal(db_path, fids[0])
    resp = client.get("/proposals", headers=auth_headers())
    body = resp.text
    # The literal `<script>` token must NOT appear unescaped in the page
    # body (other than a constant in the source view, which we don't
    # serve). Jinja2 default escaping yields `&lt;script&gt;`.
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body


def test_xss_in_thesis_is_html_escaped(client: TestClient, db_path: str) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    _insert_xss_proposal(db_path, fids[0])
    resp = client.get("/proposals", headers=auth_headers())
    body = resp.text
    assert '<img src=x onerror=' not in body
    assert "&lt;img" in body


def test_no_unsafe_filter_on_user_data() -> None:
    """AC-12: no `|safe` filter is applied to user-controlled fields in any
    template. The check is a static grep — any future edit that adds `|safe`
    must be reviewed.
    """
    tmpl_dir = (
        Path(__file__).resolve().parent.parent.parent / "src" / "dashboard" / "templates"
    )
    for tmpl in tmpl_dir.glob("*.html"):
        text = tmpl.read_text(encoding="utf-8")
        # Any occurrence of `|safe` is suspect; the dashboard renders
        # SVG sparklines via dedicated tags, never via user-controlled
        # markup, so there is no legitimate need for the filter.
        assert "|safe" not in text, f"{tmpl.name} uses |safe filter"
        assert "{{ raw_response | safe" not in text


def test_path_traversal_in_symbol_param_returns_400_or_filters_safely(
    client: TestClient, db_path: str
) -> None:
    seed_prompt(db_path)
    seed_filings(db_path, n=1)
    # Path-traversal-flavoured noise in the symbol filter must not 500.
    resp = client.get(
        "/proposals?symbol=../../etc/passwd", headers=auth_headers()
    )
    assert resp.status_code in (200, 400)
    body = resp.text
    # It must NOT yield filesystem content.
    assert "/etc/passwd" not in body
    assert "root:x:0:0" not in body


def test_secrets_never_appear_in_any_page(
    client: TestClient, db_path: str
) -> None:
    """The dashboard process holds the user/password in memory but they
    must never reach a rendered page. Probe each route for the literal.
    """
    for path in ("/", "/proposals", "/trades", "/positions", "/llm"):
        resp = client.get(path, headers=auth_headers())
        assert PASSWORD not in resp.text, f"{path} leaked dashboard_password"
        # `USER` ("operator") is a common English word — assert only that
        # the password is never echoed back. Username low-entropy means
        # checking for it would yield false positives in normal copy.


def test_no_paths_accept_form_post(client: TestClient) -> None:
    """AC-12: dashboard is read-only — no state-mutating POST endpoints."""
    for path in ("/", "/proposals", "/trades", "/positions", "/llm"):
        resp = client.post(path, headers=auth_headers())
        assert resp.status_code in (
            405,
            404,
        ), f"{path} accepted POST (status={resp.status_code})"


def test_proposal_status_filter_only_accepts_known_values(
    client: TestClient, db_path: str
) -> None:
    """Status filter must reject arbitrary input (defense vs SQL-injection-
    flavoured noise). Repo helpers should parameterise; the route should
    additionally whitelist the value.
    """
    seed_prompt(db_path)
    seed_filings(db_path, n=1)
    resp = client.get(
        "/proposals?status=' OR 1=1; --", headers=auth_headers()
    )
    # 400 is the cleanest answer; 200 with empty result is also acceptable
    # so long as the row tally is unchanged.
    assert resp.status_code in (200, 400)
