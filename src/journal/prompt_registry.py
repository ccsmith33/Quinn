"""Prompt registry — scan prompt files, hash content, upsert into `prompts`.

Architecture references: §3.1 (prompt versioning), §8.1 (prompts table),
ADR-005 (content-hash version ids), FR-29.

Per D-030: takes explicit `db_path` to match the journal-repo signature.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from observability.log_port import get_logger

from .models import PromptRow
from .repo import insert_prompt

log = get_logger(__name__)

_PROMPT_GLOBS = ("*.txt", "*.json")
_HASH_PREFIX_LEN = 12  # ADR-005 display convention


def _scan(prompt_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _PROMPT_GLOBS:
        files.extend(prompt_dir.rglob(pattern))
    return sorted(files)


def _build_row(file_path: Path) -> PromptRow:
    content = file_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    name = file_path.stem
    prompt_version = f"{name}@{digest[:_HASH_PREFIX_LEN]}"
    return PromptRow(
        prompt_version=prompt_version,
        name=name,
        file_path=str(file_path),
        content_hash=digest,
    )


def register_prompts(prompt_dir: Path, db_path: str) -> list[PromptRow]:
    """Scan `prompt_dir` for *.txt/*.json, compute version ids, upsert to `prompts`.

    Idempotent: re-registering an unchanged file is a no-op. Concurrency-safe:
    uniqueness violations resolve to "already-present".

    Returns the list of `PromptRow` objects observed on disk (one per file),
    regardless of whether each was newly inserted or already present.
    """
    rows: list[PromptRow] = []
    for path in _scan(prompt_dir):
        row = _build_row(path)
        rows.append(row)
        try:
            insert_prompt(db_path, row)
        except sqlite3.IntegrityError:
            # Primary-key collision on prompt_version → already registered.
            log.debug(
                "prompt already registered",
                extra={"prompt_version": row.prompt_version},
            )
    return rows
