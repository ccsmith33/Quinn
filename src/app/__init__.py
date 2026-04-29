"""S5.6 — agent process package.

Public surface (re-exported lazily so importing `app.state` alone in
unit tests doesn't drag in the full composition dep graph):

- `AgentState` (state machine enum)
- `AgentLoop` (the long-running coordinator)
- `compose_agent` (dependency wiring)
- `main` (process entrypoint)

`python -m src.app` runs `src/app/__main__.py`, which calls `main()`
below — the entrypoint declared in S8.3's `quinn-agent.service` unit.

The story's dev-notes mention `src/app.py` as the entrypoint; the
implementation chose a package layout (with this `__init__.py` +
sibling modules) to keep the wiring modular. `python -m src.app` works
identically either way.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .state import AgentState, IllegalAgentTransition, Transitions

# Default prompt + similarity-artifact paths; overridden via env for tests.
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_DEFAULT_SIMILARITY_ARTIFACT_DIR = "/var/lib/quinn/similarity"


def main() -> int:
    """Process entrypoint. Returns 0 on graceful shutdown, non-zero on
    fatal boot error or hard-shutdown timeout (AC-1 / AC-6).
    """
    import asyncio

    from config.loader import ConfigError, load_config
    from config.secrets import MissingSecret, load_secrets
    from journal.migrate import SchemaMismatch, apply_migrations, verify_schema
    from journal.repo import JournalRepo
    from observability.log import configure_logging

    # Step 1 — config + secrets.
    try:
        config = load_config()
    except ConfigError as e:
        print(f"FATAL: config load failed: {e}", file=sys.stderr)
        return 2
    try:
        secrets = load_secrets()
    except MissingSecret as e:
        print(f"FATAL: secrets load failed: {e}", file=sys.stderr)
        return 2

    # Step 2 — migrations + verify.
    db_path = os.environ.get("QUINN_DB_PATH", "/var/lib/quinn/journal.db")
    try:
        apply_migrations(db_path)
        verify_schema(db_path)
    except SchemaMismatch as e:
        print(f"FATAL: schema mismatch: {e}", file=sys.stderr)
        return 2

    # Step 3 — logging.
    log_level_name = config.observability.log_level
    log_level = getattr(logging, log_level_name, logging.INFO)
    configure_logging(level=log_level, secrets=secrets)

    # Step 4 — compose.
    from .composition import compose_agent

    prompts_dir = Path(
        os.environ.get("QUINN_PROMPTS_DIR", str(_DEFAULT_PROMPTS_DIR))
    )
    similarity_artifact_dir = os.environ.get(
        "QUINN_SIMILARITY_ARTIFACT_DIR", _DEFAULT_SIMILARITY_ARTIFACT_DIR
    )
    journal = JournalRepo(db_path)
    loop_obj = compose_agent(
        config,
        secrets=secrets,
        journal=journal,
        prompts_dir=prompts_dir,
        similarity_artifact_dir=similarity_artifact_dir,
    )

    # Step 5 — run.
    return asyncio.run(loop_obj.run())


__all__ = [
    "AgentState",
    "IllegalAgentTransition",
    "Transitions",
    "main",
]
