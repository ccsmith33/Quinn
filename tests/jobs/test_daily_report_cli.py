"""S8.2 follow-up — CLI shim tests for `python -m jobs.daily_report`.

The systemd unit `ops/systemd/quinn-daily-report.service` drives the daily
roll-up via `python -m jobs.daily_report`. These tests exercise the
`cli_main(...)` entrypoint with an in-process fake `Notifier`, avoiding
any real Telegram egress from the test path.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "journal.db")
    apply_migrations(path)
    return path


class _FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)


class _BrokenNotifier:
    def send(self, text: str) -> None:
        raise RuntimeError("simulated telegram outage")


def test_daily_report_cli_runs_compose_and_send_and_returns_zero(db: str) -> None:
    """Empty journal is fine — `DailyReporter.compose` returns a Report
    with zero counts; the notifier still receives the formatted text."""
    from jobs.daily_report import cli_main

    notifier = _FakeNotifier()
    rc = cli_main(
        argv=["--db", db],
        telegram=notifier,
        clock=lambda: dt.datetime(2026, 4, 28, 20, 30, tzinfo=dt.UTC),
    )
    assert rc == 0
    assert len(notifier.sent) == 1
    # Sanity: the dispatched payload is the formatted report header.
    assert "Daily report" in notifier.sent[0]


def test_daily_report_cli_returns_nonzero_on_send_failure(db: str) -> None:
    """Telegram send failure → non-zero exit so systemd surfaces it."""
    from jobs.daily_report import cli_main

    rc = cli_main(
        argv=["--db", db],
        telegram=_BrokenNotifier(),
        clock=lambda: dt.datetime(2026, 4, 28, 20, 30, tzinfo=dt.UTC),
    )
    assert rc != 0


def test_daily_report_cli_help_does_not_send(db: str) -> None:
    """`--help` exits cleanly without touching the notifier or journal."""
    from jobs.daily_report import cli_main

    notifier = _FakeNotifier()
    with pytest.raises(SystemExit) as exc:
        cli_main(
            argv=["--help"],
            telegram=notifier,
            clock=lambda: dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
        )
    assert exc.value.code == 0
    assert notifier.sent == []
