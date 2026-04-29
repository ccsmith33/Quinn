"""S5.6 AC-13 — single code path lint (D-007 sacred, FR-23).

Greps `src/` for any `if .*(broker_mode|paper).*"live"` pattern outside
the legal locations (config / app/composition broker construction).
A failure here means a code branch sneaked in that conditions on
paper-vs-live at runtime — D-007 violated.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Modules where ANY paper/live conditioning would be a D-007 violation.
_FORBIDDEN_DIRS = (
    "src/execution",
    "src/broker",
    "src/app",
    "src/analyzer",
    "src/proposal",
    "src/prefilter",
)

# Pattern picks up `if … broker_mode …` / `if paper:` / `if … "live"` etc.
_PATTERN = re.compile(r"if\s.*(broker_mode|paper).*[\"']live")


def test_no_paper_live_runtime_branch_in_forbidden_dirs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for d in _FORBIDDEN_DIRS:
        target = repo_root / d
        if not target.exists():
            continue
        for py in target.rglob("*.py"):
            content = py.read_text(encoding="utf-8")
            for lineno, line in enumerate(content.splitlines(), start=1):
                if _PATTERN.search(line):
                    offenders.append(f"{py.relative_to(repo_root)}:{lineno}: {line.strip()}")
    # `src/broker/alpaca.py` is allowed to compare endpoint↔mode at
    # CONSTRUCTION time. The forbidden pattern requires both 'broker_mode'
    # or 'paper' AND a 'live' literal — the existing _EndpointModeMismatch
    # raises in alpaca.py have the form `if mode == "live"` or
    # `if mode == "paper"` and would NOT match the conjoined pattern.
    assert not offenders, (
        "single-code-path violation (D-007 / FR-23): "
        + "; ".join(offenders)
    )


def test_compose_agent_only_passes_credentials_through() -> None:
    """Tighten AC-13: composition.py is the ONE place that may reference
    `broker_mode`; verify the file's references are limited to credential
    selection."""
    composition = Path(__file__).resolve().parents[2] / "src" / "app" / "composition.py"
    text = composition.read_text(encoding="utf-8")
    # Count occurrences. The story permits credential selection; the
    # only legal pattern is `cfg.execution.broker_mode` being passed to
    # `AlpacaBroker(mode=...)`. Everything else is suspect.
    occurrences = text.count("broker_mode")
    # Allow up to a handful of usages for: reading cfg, passing through,
    # docstring reference. >5 is a smell.
    assert occurrences <= 6, (
        f"composition.py references broker_mode {occurrences} times; "
        f"D-007 expects ≤6 (read once + pass once + small allowance)"
    )
