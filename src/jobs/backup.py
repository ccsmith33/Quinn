"""S8.3 — Daily SQLite backup to B2 + weekly restore-test (FR-31, NFR-7).

Architecture references: §9.6 Backups, §10.1 Observability (alerts).

Design:

- Backup uses SQLite's online backup API (`Connection.backup`) — atomic
  against WAL, never blocks writers.
- The backup blob is gzipped, SHA-256'd, and uploaded via a small
  `B2Uploader` Protocol. Production wires `b2sdk`'s
  `B2Api`/`Bucket.upload_local_file`; tests inject a dir-backed fake.
- Every backup attempt — success OR failure — inserts a `backups` row
  so the AlertWatcher (S8.2) can detect upload failures and stale
  backups via the same query path.
- The weekly restore-test downloads the most recent B2 object,
  decompresses it, opens it as a fresh on-disk SQLite, runs
  `PRAGMA integrity_check` and a `SELECT count(*) FROM filings` smoke
  query. Pass / fail is recorded into the LATEST `backups` row's
  `verify_result` column via a journal write.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from journal.models import BackupRow
from journal.repo import JournalRepo, connect, insert_backup
from observability.log_port import get_logger

log = get_logger(__name__)

B2_PREFIX = "quinn/journal"


class BackupUploadError(Exception):
    """Raised when the upload to B2 fails after our own retries."""


class B2Uploader(Protocol):
    """Tiny abstraction so production can wire `b2sdk` and tests can
    swap in a dir-backed fake. Three operations the backup + restore
    paths need: upload, download, latest-key-by-prefix."""

    def upload(self, *, key: str, body: bytes) -> None: ...
    def download(self, *, key: str) -> bytes: ...
    def latest_key(self, prefix: str) -> str | None: ...


@dataclass(frozen=True)
class BackupResult:
    backup_path: str  # b2://bucket/key
    sha256: str
    db_size_bytes: int
    started_at: _dt.datetime
    completed_at: _dt.datetime


@dataclass(frozen=True)
class RestoreTestResult:
    ok: bool
    detail: str  # "ok:N filings" on success; "failed:<reason>" on failure


# ---------------------------------------------------------------------------
# Backup (FR-31, AC-1)
# ---------------------------------------------------------------------------


def run_backup(
    *,
    now: _dt.datetime,
    journal: JournalRepo,
    uploader: B2Uploader,
    bucket: str,
) -> BackupResult:
    """Atomically copy `journal.db` via `sqlite3.Connection.backup`, gzip,
    SHA-256, upload to B2, insert a `backups` row.

    Raises `BackupUploadError` if the upload step fails — but ONLY after
    a `backups` row has been inserted recording the failure, so
    AlertWatcher (S8.2) still surfaces the problem.
    """
    started_at = now
    key = _b2_key_for(now)
    full_path = f"b2://{bucket}/{key}"

    with tempfile.TemporaryDirectory(prefix="quinn-backup-") as tmpdir:
        tmp_db = Path(tmpdir) / "journal.db"
        # Online backup is the SQLite-blessed atomic copy. Holds a shared
        # lock; readers/writers proceed in parallel under WAL.
        _online_backup(src_path=journal.db_path, dst_path=str(tmp_db))

        raw_size = tmp_db.stat().st_size
        gz_path = Path(tmpdir) / "journal.db.gz"
        with tmp_db.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
            dst.write(src.read())

        body = gz_path.read_bytes()
        sha = hashlib.sha256(body).hexdigest()

        try:
            uploader.upload(key=key, body=body)
        except Exception as exc:
            # Record the failed attempt and re-raise. AlertWatcher's
            # backup-failure check picks this up via verify_result LIKE
            # 'failed%'.
            insert_backup(
                journal.db_path,
                BackupRow(
                    started_at=started_at,
                    completed_at=now,
                    backup_path=full_path,
                    db_size_bytes=raw_size,
                    sha256=sha,
                    verified_at=None,
                    verify_result=f"failed:upload_{type(exc).__name__}",
                    notes=str(exc)[:500],
                ),
            )
            log.error(
                "backup.upload_failed",
                extra={"event": "backup.upload_failed", "key": key, "error": str(exc)},
            )
            if isinstance(exc, BackupUploadError):
                raise
            raise BackupUploadError(str(exc)) from exc

        completed_at = now
        insert_backup(
            journal.db_path,
            BackupRow(
                started_at=started_at,
                completed_at=completed_at,
                backup_path=full_path,
                db_size_bytes=raw_size,
                sha256=sha,
                verified_at=None,
                verify_result=None,  # set by run_restore_test
                notes=None,
            ),
        )
        log.info(
            "backup.uploaded",
            extra={
                "event": "backup.uploaded",
                "key": key,
                "size_bytes": raw_size,
                "gz_bytes": len(body),
                "sha256": sha,
            },
        )
        return BackupResult(
            backup_path=full_path,
            sha256=sha,
            db_size_bytes=raw_size,
            started_at=started_at,
            completed_at=completed_at,
        )


def _online_backup(*, src_path: str, dst_path: str) -> None:
    """Run SQLite's online backup API: source → destination, page by page,
    safe under concurrent reads/writes (WAL mode invariant).
    """
    src = sqlite3.connect(src_path, timeout=30.0)
    try:
        dst = sqlite3.connect(dst_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _b2_key_for(now: _dt.datetime) -> str:
    """`quinn/journal/YYYY/MM/DD-journal.db.gz` — S3-style hierarchical
    layout so an operator can list-and-restore by date."""
    return f"{B2_PREFIX}/{now:%Y}/{now:%m}/{now:%d}-journal.db.gz"


# ---------------------------------------------------------------------------
# Restore-test (AC-2)
# ---------------------------------------------------------------------------


def run_restore_test(
    *,
    now: _dt.datetime,
    journal: JournalRepo,
    uploader: B2Uploader,
    bucket: str,
) -> RestoreTestResult:
    """Download the most recent backup from B2; integrity-check it; record
    pass/fail on the latest `backups` row's `verify_result`.

    The `bucket` parameter is part of the production contract (used to
    build the human-readable URL written into the journal note); the
    uploader interface itself addresses objects by key, not bucket.
    """
    latest_key = uploader.latest_key(B2_PREFIX)
    if latest_key is None:
        result = RestoreTestResult(ok=False, detail="failed:no backup in B2")
        _record_verify_result(journal, now, result)
        log.warning(
            "backup.restore_test_no_backup",
            extra={"event": "backup.restore_test_no_backup", "bucket": bucket},
        )
        return result

    try:
        body = uploader.download(key=latest_key)
        decompressed = gzip.decompress(body)
    except Exception as exc:
        result = RestoreTestResult(ok=False, detail=f"failed:decompress_{type(exc).__name__}")
        _record_verify_result(journal, now, result)
        log.warning(
            "backup.restore_test_decompress_failed",
            extra={
                "event": "backup.restore_test_decompress_failed",
                "key": latest_key,
                "error": str(exc),
            },
        )
        return result

    with tempfile.TemporaryDirectory(prefix="quinn-restore-") as tmpdir:
        restored = Path(tmpdir) / "journal.db"
        restored.write_bytes(decompressed)
        try:
            with sqlite3.connect(str(restored), timeout=10.0) as conn:
                rows = conn.execute("PRAGMA integrity_check").fetchall()
                integrity_ok = len(rows) == 1 and rows[0][0] == "ok"
                if not integrity_ok:
                    detail = f"failed:integrity_check {rows!r}"
                else:
                    n = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
                    detail = f"ok:{n} filings"
        except Exception as exc:
            result = RestoreTestResult(ok=False, detail=f"failed:smoke_{type(exc).__name__}_{exc}")
            _record_verify_result(journal, now, result)
            log.warning(
                "backup.restore_test_smoke_failed",
                extra={
                    "event": "backup.restore_test_smoke_failed",
                    "key": latest_key,
                    "error": str(exc),
                },
            )
            return result

    ok = detail.startswith("ok:")
    result = RestoreTestResult(ok=ok, detail=detail)
    _record_verify_result(journal, now, result)
    log.info(
        "backup.restore_test_done",
        extra={
            "event": "backup.restore_test_done",
            "key": latest_key,
            "ok": ok,
            "detail": detail,
        },
    )
    return result


def _record_verify_result(
    journal: JournalRepo, now: _dt.datetime, result: RestoreTestResult
) -> None:
    """Update the most recent `backups` row's `verify_result` + `verified_at`.

    The `backups` table is normally append-only (NFR-16), but the
    verify columns are explicitly designed for in-place updates: the
    verify-test is a SECOND event in the same logical record's
    lifecycle. This is the one mutation the journal allows on this
    table, intentionally.
    """
    verify = result.detail
    with connect(journal.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE backups SET verify_result = ?, verified_at = ? "
            "WHERE id = (SELECT id FROM backups ORDER BY started_at DESC LIMIT 1)",
            (verify, now),
        )
        conn.execute("COMMIT")


# ---------------------------------------------------------------------------
# CLI shim — `python -m jobs.backup` (S8.3 follow-up; runbook step 10 +
# `make backup-now`).
# ---------------------------------------------------------------------------


_DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"


def cli_main(
    *,
    argv: list[str] | None = None,
    uploader: B2Uploader | None = None,
    clock: Callable[[], _dt.datetime] | None = None,
) -> int:
    """Entrypoint for `python -m jobs.backup`. Returns the process exit
    code (0 = success, non-zero on any failure).

    Test seam: `uploader` and `clock` are injectable so tests can drive
    the CLI without a real B2 account or wallclock dependency.
    Production calls with `uploader=None`, which lazy-builds a real
    `b2sdk`-backed uploader from `Secrets.backup_b2_*`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="jobs.backup",
        description="Run a one-shot SQLite → B2 backup.",
    )
    parser.add_argument("--db", default=_DEFAULT_DB_PATH)
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args(argv)

    now = (clock or (lambda: _dt.datetime.now(_dt.UTC)))()
    journal = JournalRepo(args.db)
    bucket, real_uploader = _resolve_uploader(uploader, args.bucket)
    try:
        run_backup(now=now, journal=journal, uploader=real_uploader, bucket=bucket)
    except BackupUploadError as exc:
        log.error(
            "backup.cli_failed",
            extra={"event": "backup.cli_failed", "error": str(exc)},
        )
        return 2
    return 0


def _resolve_uploader(
    uploader: B2Uploader | None, bucket_arg: str | None
) -> tuple[str, B2Uploader]:
    """Return (bucket, uploader). Test path injects both; production path
    pulls from `Secrets.backup_b2_*` and constructs a `b2sdk`-backed
    uploader. The b2sdk import is lazy so module import (and the test
    suite) does not require the package to be installed."""
    if uploader is not None:
        if bucket_arg is None:
            raise SystemExit("--bucket is required when an injected uploader is used")
        return bucket_arg, uploader

    from config.secrets import load_secrets

    secrets = load_secrets()
    bucket = bucket_arg or secrets.backup_b2_bucket.get_secret_value()
    return bucket, _build_b2sdk_uploader(
        key_id=secrets.backup_b2_key_id.get_secret_value(),
        application_key=secrets.backup_b2_application_key.get_secret_value(),
        bucket=bucket,
    )


def _build_b2sdk_uploader(*, key_id: str, application_key: str, bucket: str) -> B2Uploader:
    """Lazy-import b2sdk; build a thin adapter conforming to the
    `B2Uploader` Protocol."""
    from b2sdk.v2 import B2Api, InMemoryAccountInfo  # type: ignore[import-not-found]

    info = InMemoryAccountInfo()
    api = B2Api(info)
    api.authorize_account("production", key_id, application_key)
    b2_bucket = api.get_bucket_by_name(bucket)

    class _Real:
        def upload(self, *, key: str, body: bytes) -> None:
            b2_bucket.upload_bytes(body, file_name=key)

        def download(self, *, key: str) -> bytes:
            import io

            buf = io.BytesIO()
            b2_bucket.download_file_by_name(key).save(buf)
            return buf.getvalue()

        def latest_key(self, prefix: str) -> str | None:
            names = [f.file_name for f, _ in b2_bucket.ls(folder_to_list=prefix, recursive=True)]
            return sorted(names)[-1] if names else None

    return _Real()


if __name__ == "__main__":  # pragma: no cover - exercised via cli_main tests
    import sys

    sys.exit(cli_main())
