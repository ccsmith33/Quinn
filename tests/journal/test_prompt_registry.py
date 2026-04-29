"""S1.5 — Prompts registry tests.

Architecture references: §3.1 (prompt versioning), §8.1 (prompts table),
ADR-005, FR-29.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.prompt_registry import register_prompts
from journal.repo import connect, get_prompt_by_version


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


def _expected_version(name: str, content: bytes) -> str:
    return f"{name}@{hashlib.sha256(content).hexdigest()[:12]}"


def _count_prompts(db_path: str) -> int:
    with connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0])


def test_register_creates_rows(tmp_path: Path, db: str) -> None:
    """AC-1: walks dir, computes content-hash version ids, inserts rows."""
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    a = pdir / "sonnet_filing_analysis_v1.txt"
    a.write_text("alpha prompt body")
    b = pdir / "opus_proposal_review_v1.txt"
    b.write_text("beta prompt body")

    rows = register_prompts(pdir, db)

    assert len(rows) == 2
    versions = {r.prompt_version for r in rows}
    assert _expected_version("sonnet_filing_analysis_v1", a.read_bytes()) in versions
    assert _expected_version("opus_proposal_review_v1", b.read_bytes()) in versions
    assert _count_prompts(db) == 2

    # round-trip via repo
    fetched = get_prompt_by_version(
        db, _expected_version("sonnet_filing_analysis_v1", a.read_bytes())
    )
    assert fetched is not None
    assert fetched.name == "sonnet_filing_analysis_v1"


def test_idempotent_on_unchanged_files(tmp_path: Path, db: str) -> None:
    """AC-1 idempotent: second call with unchanged files does not duplicate rows."""
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "p1.txt").write_text("body one")
    (pdir / "p2.txt").write_text("body two")

    register_prompts(pdir, db)
    register_prompts(pdir, db)

    assert _count_prompts(db) == 2


def test_change_creates_new_version(tmp_path: Path, db: str) -> None:
    """AC-1: editing a file produces a new version row (append-only, old preserved)."""
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    f = pdir / "p1.txt"
    f.write_text("v1 body")

    register_prompts(pdir, db)
    assert _count_prompts(db) == 1

    f.write_text("v2 body — edited")
    register_prompts(pdir, db)
    assert _count_prompts(db) == 2  # old version preserved, new version added


def test_empty_dir_no_error(tmp_path: Path, db: str) -> None:
    """AC-2: empty dir returns [] without erroring."""
    pdir = tmp_path / "empty"
    pdir.mkdir()
    rows = register_prompts(pdir, db)
    assert rows == []
    assert _count_prompts(db) == 0


def test_concurrent_calls_no_crash(tmp_path: Path, db: str) -> None:
    """AC-3: simultaneous callers resolve to already-present rather than crash."""
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "p1.txt").write_text("body")
    (pdir / "p2.txt").write_text("body two")
    (pdir / "p3.txt").write_text("body three")

    errors: list[Exception] = []

    def worker() -> None:
        try:
            register_prompts(pdir, db)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert _count_prompts(db) == 3


def test_recursive_walk_includes_subdirectories(tmp_path: Path, db: str) -> None:
    """AC-1: walks `prompt_dir` recursively for *.txt and *.json."""
    pdir = tmp_path / "prompts"
    (pdir / "fragments").mkdir(parents=True)
    (pdir / "schemas").mkdir()
    (pdir / "top.txt").write_text("top body")
    (pdir / "fragments" / "frag.txt").write_text("fragment body")
    (pdir / "schemas" / "schema.json").write_text('{"k":"v"}')

    rows = register_prompts(pdir, db)
    assert len(rows) == 3
    names = {r.name for r in rows}
    assert names == {"top", "frag", "schema"}


def test_non_prompt_files_ignored(tmp_path: Path, db: str) -> None:
    """AC-1: only *.txt and *.json are registered."""
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "keep.txt").write_text("keep")
    (pdir / "keep.json").write_text("{}")
    (pdir / "skip.md").write_text("skip")
    (pdir / "skip.py").write_text("# skip")

    rows = register_prompts(pdir, db)
    assert len(rows) == 2
    assert {r.name for r in rows} == {"keep"}
    extensions = sorted(Path(r.file_path).suffix for r in rows)
    assert extensions == [".json", ".txt"]
    assert _count_prompts(db) == 2
