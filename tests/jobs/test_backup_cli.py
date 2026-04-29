"""S8.3 follow-up — CLI shim tests for `jobs.backup` and `jobs.restore_from_b2`.

The runbook + Makefile reference `python -m jobs.backup` and
`python -m jobs.restore_from_b2` to drive the operational backup/restore
paths. These tests exercise the `cli_main(...)` entrypoint with an
in-process fake `B2Uploader`, avoiding the need for real B2 credentials
or the `b2sdk` package in the test path.
"""

from __future__ import annotations

import datetime as dt
import gzip
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import FilingRow
from journal.repo import insert_filing


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "journal.db")
    apply_migrations(path)
    return path


@pytest.fixture
def b2_dir(tmp_path: Path) -> Path:
    d = tmp_path / "b2"
    d.mkdir()
    return d


class _FakeUploader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def upload(self, *, key: str, body: bytes) -> None:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

    def download(self, *, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def latest_key(self, prefix: str) -> str | None:
        files = sorted(p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file())
        matches = [str(p) for p in files if str(p).startswith(prefix)]
        return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# `python -m jobs.backup` shim
# ---------------------------------------------------------------------------


def test_backup_cli_main_runs_backup_and_returns_zero(db: str, b2_dir: Path) -> None:
    """The CLI shim builds a JournalRepo + uploader, invokes run_backup,
    and returns 0 on success."""
    from jobs.backup import cli_main

    rc = cli_main(
        argv=[
            "--db",
            db,
            "--bucket",
            "test-bucket",
        ],
        uploader=_FakeUploader(b2_dir),
        clock=lambda: dt.datetime(2026, 4, 28, 6, 0, 0, tzinfo=dt.UTC),
    )
    assert rc == 0
    # B2 path layout same as run_backup contract.
    assert (b2_dir / "quinn/journal/2026/04/28-journal.db.gz").is_file()


def test_backup_cli_main_returns_nonzero_on_upload_failure(db: str, b2_dir: Path) -> None:
    """Upload failure → non-zero exit. The runbook step 10 "smoke" probe
    relies on this so an operator-driven `make backup-now` surfaces the
    failure as an exit code, not silent success."""
    from jobs.backup import BackupUploadError, cli_main

    class _Broken:
        def upload(self, *, key: str, body: bytes) -> None:
            raise BackupUploadError("simulated outage")

        def download(self, *, key: str) -> bytes:
            raise NotImplementedError

        def latest_key(self, prefix: str) -> str | None:
            return None

    rc = cli_main(
        argv=["--db", db, "--bucket", "b"],
        uploader=_Broken(),
        clock=lambda: dt.datetime(2026, 4, 28, 6, 0, 0, tzinfo=dt.UTC),
    )
    assert rc != 0


def test_backup_cli_main_help_does_not_run_backup(db: str, b2_dir: Path) -> None:
    """`python -m jobs.backup --help` must print help + exit cleanly without
    touching the journal."""
    from jobs.backup import cli_main

    with pytest.raises(SystemExit) as exc:
        cli_main(
            argv=["--help"],
            uploader=_FakeUploader(b2_dir),
            clock=lambda: dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
        )
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# `python -m jobs.restore_from_b2` — rehydration restore (NFR-7 step 6)
# ---------------------------------------------------------------------------


def test_restore_cli_writes_target_path(db: str, b2_dir: Path, tmp_path: Path) -> None:
    """Rehydration restore: download latest backup → decompress → write
    to `--target` path. The runbook's step 6 calls this to repopulate
    `/var/lib/quinn/journal.db` after fresh droplet provisioning."""
    from jobs.backup import run_backup
    from jobs.restore_from_b2 import cli_main as restore_main

    # Seed a filing so the restored DB has something to verify against.
    insert_filing(
        db,
        FilingRow(
            accession_number="0001-01",
            cik=1,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
            fetched_at=dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
            raw_text_path="/dev/null",
            content_hash="x" * 64,
        ),
    )

    uploader = _FakeUploader(b2_dir)
    from journal.repo import JournalRepo

    run_backup(
        now=dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
        journal=JournalRepo(db),
        uploader=uploader,
        bucket="b",
    )

    target = tmp_path / "restored.db"
    rc = restore_main(
        argv=["--target", str(target), "--bucket", "b"],
        uploader=uploader,
    )
    assert rc == 0
    assert target.is_file()

    # The restored DB carries the seeded filing.
    with sqlite3.connect(str(target)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    assert n == 1, "restored DB lost the seeded filing"


def test_restore_cli_returns_nonzero_when_no_backup(b2_dir: Path, tmp_path: Path) -> None:
    """No backup in B2 → non-zero exit. Aborts the rehydration runbook
    so the operator notices before continuing past step 6."""
    from jobs.restore_from_b2 import cli_main as restore_main

    target = tmp_path / "restored.db"
    rc = restore_main(
        argv=["--target", str(target), "--bucket", "b"],
        uploader=_FakeUploader(b2_dir),
    )
    assert rc != 0
    assert not target.is_file()


def test_restore_cli_decompresses_correctly(db: str, b2_dir: Path, tmp_path: Path) -> None:
    """Sanity: the restored file is the SQLite source bytes, not the gzip
    blob. Operator running `make verify-schema` would otherwise hit a
    'file is not a database' error."""
    from jobs.backup import run_backup
    from jobs.restore_from_b2 import cli_main as restore_main
    from journal.repo import JournalRepo

    uploader = _FakeUploader(b2_dir)
    run_backup(
        now=dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
        journal=JournalRepo(db),
        uploader=uploader,
        bucket="b",
    )

    target = tmp_path / "restored.db"
    restore_main(
        argv=["--target", str(target), "--bucket", "b"],
        uploader=uploader,
    )

    head = target.read_bytes()[:16]
    assert head.startswith(b"SQLite format 3"), (
        f"restored file is not a SQLite db; first bytes: {head!r}"
    )
    # Confirm it's NOT still gzip-compressed
    with pytest.raises((OSError, gzip.BadGzipFile)):
        gzip.decompress(target.read_bytes()[:64])
