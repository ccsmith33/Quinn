#!/usr/bin/env bash
# S9.2 AC-6 (D-064) — verify the Quinn git remote is reachable.
#
# Runs three read-only probes:
#   1. `git remote -v` — confirms an `origin` remote is configured.
#   2. `git fetch origin --dry-run` — confirms the remote is reachable.
#   3. `gh repo view --web 2>/dev/null || ...` — opens the repo in a
#      browser if `gh` is logged in; otherwise prints a one-line hint.
#
# Exits 0 on success, non-zero if either of the first two probes fails.

set -eu

if ! command -v git >/dev/null 2>&1; then
    echo "verify-remote: git is not installed" >&2
    exit 1
fi

echo "== git remote =="
git remote -v

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "verify-remote: no 'origin' remote configured" >&2
    echo "verify-remote: run \`git remote add origin <repo-url>\` first" >&2
    exit 2
fi

echo
echo "== git fetch (dry-run) =="
if ! git fetch origin --dry-run; then
    echo "verify-remote: fetch failed; check network / credentials" >&2
    exit 3
fi

echo
echo "== gh repo view =="
if command -v gh >/dev/null 2>&1; then
    # Prefer the in-terminal view; fall back to a plain message if `gh`
    # is unauthenticated or the repo is unknown to it.
    gh repo view 2>/dev/null || echo "open the repo manually (gh not authenticated or repo unknown)"
else
    echo "gh is not installed; skip"
fi

echo
echo "verify-remote: OK"
