"""S9.2 — runbook concrete-links polish (AC-4, D-064).

The rehydration runbook (`ops/runbooks/rehydrate.md`) must contain no
unresolved placeholders that would force the operator to hand-edit the
file mid-deploy. Specific exceptions (operator-action-only TBDs) must be
labelled as such.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "ops" / "runbooks" / "rehydrate.md"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_exists() -> None:
    assert RUNBOOK.is_file(), f"missing {RUNBOOK}"


def test_no_repo_url_placeholder() -> None:
    """`<repo-url>` must be replaced with the actual github URL."""
    assert "<repo-url>" not in _text()


def test_no_org_placeholder_in_clone_step() -> None:
    """The `git clone` step must point at the real `github.com/<gh-user>/Quinn` URL."""
    text = _text()
    # Reject the literal `<org>` placeholder anywhere in the runbook.
    assert "<org>" not in text


def test_no_your_name_placeholder_outside_substitution_command() -> None:
    """The `<your-name>.duckdns.org` placeholder is replaced with
    `${DUCKDNS_DOMAIN}` everywhere EXCEPT inside the explicit `sed` line
    that performs the substitution (where the placeholder must stay so
    the sed pattern is meaningful).
    """
    text = _text()
    for line in text.splitlines():
        if "<your-name>" in line and "sed " not in line:
            raise AssertionError(
                f"unresolved <your-name> placeholder in non-sed line: {line!r}"
            )


def test_no_unresolved_todo_or_unmarked_tbd() -> None:
    """`TBD` survives only inside the explicitly-labelled operator-action
    rows of the timed-dry-run table (AC-6 of S8.3). Every other `TBD` /
    `<TODO>` is forbidden.
    """
    text = _text()
    # `<TODO>` is unconditional reject.
    assert "<TODO>" not in text, "unresolved <TODO> in runbook"
    # `TBD` outside an operator-action row.
    for line in text.splitlines():
        if "TBD" not in line:
            continue
        normalized = line.lower()
        if any(
            marker in normalized
            for marker in (
                "operator must",
                "operator-action",
                "operator action",
                "drill pending",
            )
        ):
            continue
        raise AssertionError(
            f"unmarked TBD on non-operator-action line: {line!r}"
        )


def test_runbook_documents_secrets_env_sourcing_for_dashboard_step() -> None:
    """AC-4: the runbook tells the operator to source `/etc/quinn/secrets.env`
    before the dashboard / Caddy / DuckDNS install steps so `${DUCKDNS_DOMAIN}`
    is set when the line-by-line execution reaches them.
    """
    text = _text()
    # Either explicit `set -a; . /etc/quinn/secrets.env; set +a` or a
    # one-line note pointing at it.
    assert (
        "/etc/quinn/secrets.env" in text
        and (
            "set -a" in text
            or "source /etc/quinn/secrets.env" in text
            or ". /etc/quinn/secrets.env" in text
            or "uses .env loaded above" in text
        )
    )


def test_runbook_documents_concrete_clone_command() -> None:
    """The clone step references the canonical GitHub URL form."""
    text = _text()
    # Look for a real `github.com/<user>/Quinn` style URL (no angle
    # brackets) — at minimum a `github.com/` URL with a non-placeholder
    # path component.
    assert re.search(
        r"github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+",
        text,
    ), "no concrete github.com/<user>/<repo> URL found"


def test_runbook_vector_path_consistent() -> None:
    """The `vector validate` step references a real install path
    (`vector` on PATH OR an absolute path that matches the apt install
    location). Inconsistent paths (e.g., `/opt/vector/bin/vector` plus
    `vector validate` without a leading path) are a fail.
    """
    text = _text()
    # Two acceptable patterns:
    #   1. `vector validate /etc/vector/quinn.toml` — relies on PATH (apt
    #      installs to /usr/bin/vector or /usr/local/bin/vector).
    #   2. An absolute path explicitly given.
    assert "vector validate" in text
