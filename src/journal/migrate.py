"""SQLite migration runner for Quinn v1.

Architecture references: §8.3 (numbered SQL files, no Alembic — D-023),
§2.9 (process refuses to start on schema-version mismatch).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_FILENAME_RE = re.compile(r"^(\d+)_.+\.sql$")


class SchemaMismatch(Exception):
    """Raised when on-disk migration files do not match meta-recorded version."""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _strip_sql_comments(sql: str) -> str:
    out_lines: list[str] = []
    for line in sql.splitlines():
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def _split_sql_statements(sql: str) -> list[str]:
    cleaned = _strip_sql_comments(sql)
    return [s.strip() for s in cleaned.split(";") if s.strip()]


def _discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """Return [(version, path)] sorted lexically, validating the filename pattern."""
    files = sorted(migrations_dir.glob("*.sql"))
    discovered: list[tuple[int, Path]] = []
    for f in files:
        m = _FILENAME_RE.match(f.name)
        if not m:
            raise SchemaMismatch(f"Migration filename does not match NNN_*.sql: {f.name}")
        discovered.append((int(m.group(1)), f))
    return discovered


def _meta_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    return row is not None


def _applied_versions(conn: sqlite3.Connection) -> set[int]:
    if not _meta_table_exists(conn):
        return set()
    rows = conn.execute("SELECT schema_version FROM meta").fetchall()
    return {r[0] for r in rows}


def apply_migrations(db_path: str) -> int:
    """Apply any unapplied migration files in lexical order; return final schema version."""
    conn = _connect(db_path)
    try:
        discovered = _discover_migrations()
        if not discovered:
            return 0
        applied = _applied_versions(conn)
        for version, path in discovered:
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            statements = _split_sql_statements(sql)
            try:
                conn.execute("BEGIN")
                for stmt in statements:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO meta(schema_version, applied_at) "
                    "VALUES (?, CURRENT_TIMESTAMP)",
                    (version,),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return max(v for v, _ in discovered)
    finally:
        conn.close()


def verify_schema(db_path: str) -> None:
    """Raise SchemaMismatch if on-disk migration set does not match meta."""
    conn = _connect(db_path)
    try:
        if not _meta_table_exists(conn):
            raise SchemaMismatch("meta table is missing; database not initialized")
        applied = _applied_versions(conn)
        on_disk = {v for v, _ in _discover_migrations()}
        if applied != on_disk:
            missing_from_db = on_disk - applied
            extra_in_db = applied - on_disk
            raise SchemaMismatch(
                f"schema mismatch — on-disk versions {sorted(on_disk)}, "
                f"applied versions {sorted(applied)}, "
                f"missing from db: {sorted(missing_from_db)}, "
                f"extra in db: {sorted(extra_in_db)}"
            )
    finally:
        conn.close()
