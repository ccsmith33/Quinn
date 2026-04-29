"""Rehydration restore — `python -m jobs.restore_from_b2 --target ...`.

S8.3 runbook step 6: download the most-recent B2 backup, decompress
the gzip, write the SQLite bytes to the operator-supplied target path
(typically `/var/lib/quinn/journal.db` on a fresh droplet).

This is distinct from `jobs.backup.run_restore_test` which is the
weekly verify-test. The restore-test reads the latest backup and
checks integrity in-memory. This module writes the latest backup to
disk so the agent loop can boot from it after rehydration.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

from jobs.backup import B2_PREFIX, B2Uploader, _resolve_uploader
from observability.log_port import get_logger

log = get_logger(__name__)


def restore_from_b2(*, target: Path, uploader: B2Uploader) -> bool:
    """Download the most-recent B2 backup and write it to `target`.

    Returns True on success, False if no backup is available in B2 or
    the download/decompress fails. The caller (CLI shim) maps this to
    a process exit code.
    """
    latest_key = uploader.latest_key(B2_PREFIX)
    if latest_key is None:
        log.error(
            "restore.no_backup",
            extra={"event": "restore.no_backup", "prefix": B2_PREFIX},
        )
        return False

    try:
        body = uploader.download(key=latest_key)
        decompressed = gzip.decompress(body)
    except Exception as exc:
        log.error(
            "restore.download_failed",
            extra={
                "event": "restore.download_failed",
                "key": latest_key,
                "error": str(exc),
            },
        )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(decompressed)
    log.info(
        "restore.completed",
        extra={
            "event": "restore.completed",
            "key": latest_key,
            "target": str(target),
            "bytes": len(decompressed),
        },
    )
    return True


def cli_main(
    *,
    argv: list[str] | None = None,
    uploader: B2Uploader | None = None,
) -> int:
    """Entrypoint for `python -m jobs.restore_from_b2`. Returns the
    process exit code (0 = success, non-zero on failure).

    `uploader` is injectable for tests; in production it is None and
    the secrets-backed real uploader is built on the fly.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="jobs.restore_from_b2",
        description="Download the most-recent B2 backup and write it to a target path.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Destination path for the restored SQLite db (e.g. /var/lib/quinn/journal.db)",
    )
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args(argv)

    _bucket, real_uploader = _resolve_uploader(uploader, args.bucket)
    target = Path(args.target)
    ok = restore_from_b2(target=target, uploader=real_uploader)
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover - exercised via cli_main tests
    sys.exit(cli_main())
