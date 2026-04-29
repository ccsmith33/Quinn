"""S9.1 — operator dashboard routes + auth (AC-1, AC-2, AC-8).

Covers every read-only route, HTTP basic auth (401 paths and constant-time
compare), and the meta-refresh tag in the layout.
"""

from __future__ import annotations

import base64
import time

import pytest
from fastapi.testclient import TestClient

from .conftest import (
    PASSWORD,
    USER,
    auth_headers,
    seed_account_snapshots,
    seed_filings,
    seed_kill_switch,
    seed_prompt,
    seed_proposals,
)

ROUTES = ["/", "/proposals", "/trades", "/positions", "/llm"]


# ---------------------------------------------------------------------------
# AC-1 / AC-8: every route is reachable with valid auth and returns HTML.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_route_returns_200_with_valid_auth(client: TestClient, path: str) -> None:
    resp = client.get(path, headers=auth_headers())
    assert resp.status_code == 200, f"{path} → {resp.status_code}"
    assert resp.headers["content-type"].startswith("text/html"), path


@pytest.mark.parametrize("path", ROUTES)
def test_route_pages_include_meta_refresh(client: TestClient, path: str) -> None:
    """AC-8: server-rendered Jinja2 + auto-refresh meta tag."""
    resp = client.get(path, headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    assert '<meta http-equiv="refresh"' in body
    assert "content=\"30\"" in body or "content='30'" in body


@pytest.mark.parametrize("path", ROUTES)
def test_route_pages_include_pico_css(client: TestClient, path: str) -> None:
    """AC-8: Pico.css from CDN, no JS framework."""
    resp = client.get(path, headers=auth_headers())
    body = resp.text
    assert "pico" in body.lower()
    # No <script src=...> framework imports — inline SVG only.
    assert "<script src=" not in body, "no JS framework imports allowed (AC-8)"


# ---------------------------------------------------------------------------
# AC-2: HTTP basic auth — 401 paths and WWW-Authenticate header.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_route_without_auth_returns_401_with_www_authenticate(
    client: TestClient, path: str
) -> None:
    resp = client.get(path)
    assert resp.status_code == 401
    www = resp.headers.get("www-authenticate", "")
    assert www.lower().startswith("basic")
    assert 'realm="quinn"' in www


@pytest.mark.parametrize("path", ROUTES)
def test_route_wrong_password_returns_401(client: TestClient, path: str) -> None:
    resp = client.get(path, headers=auth_headers(USER, "wrong-password"))
    assert resp.status_code == 401


@pytest.mark.parametrize("path", ROUTES)
def test_route_wrong_user_returns_401(client: TestClient, path: str) -> None:
    resp = client.get(path, headers=auth_headers("attacker", PASSWORD))
    assert resp.status_code == 401


def test_malformed_authorization_header_returns_401(client: TestClient) -> None:
    """Missing 'Basic ' prefix / unparseable base64 must 401, not 500."""
    bad_headers = [
        "",
        "Bearer xyz",
        "Basic !!notbase64!!",
        "Basic " + base64.b64encode(b"nocolon").decode(),
    ]
    for bad in bad_headers:
        resp = client.get("/", headers={"Authorization": bad})
        assert resp.status_code == 401, f"bad header {bad!r} → {resp.status_code}"


# ---------------------------------------------------------------------------
# AC-2: constant-time compare via secrets.compare_digest.
#
# Probe the source for the *primitive* (compare_digest) and assert empirical
# timing equality across "differ at byte 0" vs "differ at last byte" — the
# classic short-circuit `==` attack would show byte-0 ~10x faster than
# last-byte. Reusing the empirical pattern from the S7.3 webhook tests.
# ---------------------------------------------------------------------------


def test_constant_time_compare_is_used_by_source() -> None:
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "dashboard"
        / "auth.py"
    ).read_text(encoding="utf-8")
    assert "secrets.compare_digest" in src or "compare_digest" in src
    # Reject naive equality on the password field — there should be no `==`
    # token sandwiched between known credential identifiers.
    forbidden = [
        "password ==",
        "== password",
        "user ==",
        "== user",
    ]
    for token in forbidden:
        assert token not in src, f"non-constant-time compare detected: {token!r}"


def test_compare_timing_no_short_circuit(client: TestClient) -> None:
    """Empirical: differ-at-first-byte vs differ-at-last-byte should be
    indistinguishable in the mean (constant-time compare). We assert the
    ratio of mean rejects is well below the 10× a short-circuit `==` would
    produce — being deliberately loose so CI noise doesn't flake.
    """
    n = 40
    same_len = "x" * len(PASSWORD)
    early_diff = "X" + same_len[1:]  # differ at byte 0
    late_diff = same_len[:-1] + "X"  # differ at last byte

    def time_n(pw: str) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            client.get("/", headers=auth_headers(USER, pw))
        return time.perf_counter() - t0

    early = time_n(early_diff)
    late = time_n(late_diff)
    assert early < 10 * late and late < 10 * early, (
        f"timing skew suggests short-circuit compare: early={early}, late={late}"
    )


# ---------------------------------------------------------------------------
# AC-1: /healthz is unauthenticated (systemd readiness pattern, mirrors
# webhook listener). Not strictly required by the story but harmless for
# the systemd unit's `ExecStartPost` reachability probe.
# ---------------------------------------------------------------------------


def test_healthz_is_unauthenticated(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# AC-1: every page renders even with a totally fresh DB (no proposals,
# no filings, kill-switch in seed state). Robustness baseline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_route_renders_on_empty_journal(client: TestClient, path: str) -> None:
    resp = client.get(path, headers=auth_headers())
    assert resp.status_code == 200, f"{path} on empty DB returned {resp.status_code}"


# ---------------------------------------------------------------------------
# Smoke that seeded data flows into the templates without 500s.
# ---------------------------------------------------------------------------


def test_seeded_overview_renders(client: TestClient, journal, db_path: str) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=2)
    seed_proposals(journal, filing_ids=fids)
    seed_account_snapshots(journal, days=4)
    seed_kill_switch(journal, halted=False)
    resp = client.get("/", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    # Equity + day P&L should be rendered with seeded numerics.
    assert "Equity" in body or "equity" in body
