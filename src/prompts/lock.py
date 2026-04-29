"""prompts.lock writer/verifier — CI tripwire for unintentional prompt edits.

Per ADR-005 and D-032: lists each `*.txt` and `*.json` file under the prompt
dir with its SHA-256 hash. CI runs `verify_lock`; manual regenerate via
`write_lock`. Renaming a prompt file requires regenerating the lock.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPT_GLOBS = ("*.txt", "*.json")


class LockMismatch(Exception):
    """Raised when on-disk prompt files do not match `prompts.lock`."""


def _scan(prompt_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _PROMPT_GLOBS:
        files.extend(prompt_dir.rglob(pattern))
    return sorted(files)


def _entries(prompt_dir: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in _scan(prompt_dir):
        relpath = str(path.relative_to(prompt_dir)).replace("\\", "/")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out.append((relpath, digest))
    return out


def write_lock(prompt_dir: Path, lock_path: Path) -> None:
    """Regenerate `prompts.lock` from on-disk files."""
    lines = [f"{relpath}  {digest}" for relpath, digest in _entries(prompt_dir)]
    lock_path.write_text("\n".join(lines) + "\n")


def verify_lock(prompt_dir: Path, lock_path: Path) -> None:
    """Raise `LockMismatch` if files on disk differ from the lock."""
    if not lock_path.is_file():
        raise LockMismatch(f"{lock_path} is missing")
    expected = lock_path.read_text().strip().splitlines()
    expected_map = dict(_parse_line(line) for line in expected if line.strip())
    actual_map = dict(_entries(prompt_dir))
    if expected_map != actual_map:
        missing = sorted(set(expected_map) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected_map))
        changed = sorted(
            p for p in expected_map.keys() & actual_map.keys()
            if expected_map[p] != actual_map[p]
        )
        raise LockMismatch(
            f"prompts.lock mismatch — missing: {missing}, extra: {extra}, "
            f"changed: {changed}"
        )


def _parse_line(line: str) -> tuple[str, str]:
    parts = line.rsplit("  ", 1)
    if len(parts) != 2:
        raise LockMismatch(f"malformed lock line: {line!r}")
    return parts[0], parts[1]
