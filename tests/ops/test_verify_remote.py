"""S9.2 — `make verify-remote` / `scripts/verify_remote.sh` (AC-6, D-064)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "verify_remote.sh"
MAKEFILE = REPO / "Makefile"


def test_verify_remote_script_exists() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_verify_remote_script_is_executable() -> None:
    mode = SCRIPT.stat().st_mode
    assert mode & 0o111, f"{SCRIPT} is not executable (mode={oct(mode)})"


def test_verify_remote_script_runs_three_probes() -> None:
    """The three probes named in AC-6 must all be present in the script."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "git remote -v" in text
    assert "git fetch origin --dry-run" in text
    assert "gh repo view" in text


def test_verify_remote_makefile_target_exists() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "verify-remote:" in text
    # Phony declaration so `make verify-remote` is a real target, not a
    # filename match.
    assert "verify-remote" in text.split(".PHONY:")[1].split("\n")[0]


def test_verify_remote_script_uses_set_eu() -> None:
    """Bash strict-ish mode: `set -eu` so the script halts on the first
    failed probe rather than chaining through.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -eu" in text or "set -e" in text


def test_verify_remote_script_handles_missing_origin() -> None:
    """If no `origin` is configured, the script must exit non-zero with a
    helpful message — not silently succeed.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    assert "no 'origin' remote configured" in text
    # Some non-zero exit is enforced.
    assert "exit 2" in text or "exit 1" in text
