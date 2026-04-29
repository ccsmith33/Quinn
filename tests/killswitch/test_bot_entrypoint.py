"""S7.2 — entry-point wiring smoke tests for src/bot.py.

These do not start a real Telegram client; they exercise `build_application`
to confirm the wiring is constructible and that the entry point honours the
ADR-004 separate-process invariant (only depends on JournalRepo + KillSwitch
+ secrets — no agent-loop import).
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.repo import JournalRepo
from killswitch.api import KillSwitch
from killswitch.telegram_bot import BotCore


@pytest.fixture
def core(tmp_path: Path) -> BotCore:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    journal = JournalRepo(str(db_path))
    ks = KillSwitch(journal)
    return BotCore(ks=ks, journal=journal, allowed_chat_id=42)


def test_build_application_constructs_without_starting(core: BotCore) -> None:
    """Build a PTB Application against a dummy token; do not run polling."""
    bot_module = importlib.import_module("bot")
    app = bot_module.build_application("123456:dummy-token-for-tests", core)
    assert app is not None
    # at least one handler registered
    assert len(app.handlers) >= 1


def test_bot_module_does_not_import_agent_loop() -> None:
    """ADR-004 / AC-5: bot is a separate process with no agent-loop deps."""
    bot_src = Path(__file__).resolve().parent.parent.parent / "src" / "bot.py"
    text = bot_src.read_text(encoding="utf-8")
    forbidden = ["agent_loop", "agent.compose", "from app", "from broker"]
    for token in forbidden:
        assert token not in text, f"src/bot.py imports {token!r} (would couple crash domains)"


def test_bot_main_signature_takes_db_path() -> None:
    bot_module = importlib.import_module("bot")
    sig = inspect.signature(bot_module.main)
    assert "db_path" in sig.parameters
