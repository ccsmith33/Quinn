#!/usr/bin/env python3
"""Read-only dry-run of the Sonnet analyzer for a single filing.

Fetches a filing's existing body from disk, optionally augments it with
its EDGAR Exhibit 99.x content (the press release / supplement that the
production ingest currently strips out for 8-K Item 2.02 / 7.01 / 8.01
filings), runs it through the production Sonnet prompt (byte-identical
via `prompts.loader.PromptBuilder`), and prints the decision + reasoning.

NO database writes. NO journal side effects. NO Opus reviewer call.
This is a manual diagnostic for the question "would Sonnet propose a
trade if it had the exhibit content?" — answer it without waiting for
the next market open.

**RUN LOCALLY, NOT ON THE DROPLET.** A counterfactual `--include-exhibits`
run holds 50-150KB of augmented filing text in memory plus the Anthropic
SDK; the production droplet (1 GB RAM, no swap, ~100 MB available with
the agent running) will OOM-kill the agent. The script refuses to run
when MemAvailable < 250 MB unless `--force` is passed. Local usage:

    # from the project root, after copying the journal DB locally:
    scp root@quinn-stocks:/var/lib/quinn/journal.db ./journal-snapshot.db
    set -a; . .env; set +a
    python ops/scripts/dryrun_analyzer.py --db ./journal-snapshot.db \\
        --filing-id 487 --include-exhibits

Requires `ANTHROPIC_API_KEY` in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# Add `src/` to sys.path so we can import the production prompt builder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from journal.models import FilingRow  # noqa: E402
from prompts.loader import AnalyzerContext, PromptBuilder  # noqa: E402

EDGAR_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_DB = "/var/lib/quinn/journal.db"
DEFAULT_PROMPT_DIR = _REPO_ROOT / "src" / "prompts"
EX99_RE = re.compile(r"ex-?99[-_]?\d*", re.IGNORECASE)
SONNET_MODEL = "claude-sonnet-4-6"
DEFAULT_USER_AGENT = "Quinn-Research/v1 ccsmith33@crimson.ua.edu"
DEFAULT_EXHIBIT_CAP_BYTES = 256_000  # 256KB per exhibit; reduced from 1MB after droplet OOM
MIN_FREE_MEMORY_MB = 250  # refuse to run below this on Linux unless --force


def check_memory_or_exit(force: bool) -> None:
    """Refuse to run on a memory-tight host (e.g. the production droplet)
    unless --force is passed. Reads /proc/meminfo on Linux; no-op elsewhere.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return
    available_kb: int | None = None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
            break
    if available_kb is None:
        return
    available_mb = available_kb / 1024
    if available_mb < MIN_FREE_MEMORY_MB and not force:
        raise SystemExit(
            f"REFUSED: only {available_mb:.0f} MB available, need "
            f"{MIN_FREE_MEMORY_MB} MB. Run on a host with more RAM, or "
            f"pass --force if you accept the OOM risk. The production "
            f"droplet (1 GB total) cannot run --include-exhibits — copy "
            f"the journal DB locally and run from your dev machine instead."
        )


def fetch_filing_row(
    db_path: str, *, filing_id: int | None, accession: str | None
) -> FilingRow:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if filing_id is not None:
        cur = conn.execute("SELECT * FROM filings WHERE id = ?", (filing_id,))
    else:
        cur = conn.execute(
            "SELECT * FROM filings WHERE accession_number = ?", (accession,)
        )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(
            f"filing not found: filing_id={filing_id} accession={accession}"
        )
    return FilingRow(
        id=row["id"],
        accession_number=row["accession_number"],
        cik=row["cik"],
        form_type=row["form_type"],
        filed_at=datetime.fromisoformat(row["filed_at"]),
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        raw_text_path=row["raw_text_path"],
        content_hash=row["content_hash"],
        item_codes=row["item_codes"],
        issuer_ticker=row["issuer_ticker"],
        ingest_state=row["ingest_state"],
        ingest_error=row["ingest_error"],
    )


def edgar_get(url: str, *, user_agent: str, max_bytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read(max_bytes + 1)
    if len(body) > max_bytes:
        body = body[:max_bytes] + b"\n[...TRUNCATED AT CAP...]"
    return body


def fetch_index(cik: int, accession: str, *, user_agent: str) -> dict:
    no_dash = accession.replace("-", "")
    url = f"{EDGAR_BASE}/{cik}/{no_dash}/index.json"
    body = edgar_get(url, user_agent=user_agent, max_bytes=2_000_000)
    return json.loads(body)


def find_ex99_filenames(index: dict) -> list[str]:
    items = index.get("directory", {}).get("item", [])
    out: list[str] = []
    for it in items:
        name = it.get("name", "")
        lower = name.lower()
        if not (lower.endswith(".htm") or lower.endswith(".html")):
            continue
        if EX99_RE.search(name):
            out.append(name)
    return sorted(out)


def fetch_exhibit_text(
    cik: int, accession: str, filename: str, *, user_agent: str, cap: int
) -> str:
    no_dash = accession.replace("-", "")
    url = f"{EDGAR_BASE}/{cik}/{no_dash}/{filename}"
    body = edgar_get(url, user_agent=user_agent, max_bytes=cap)
    # Crude tag strip — sufficient for analyzer eval; production should use a
    # proper HTML→text pass.
    text = re.sub(rb"<script[^>]*>.*?</script>", b" ", body, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(rb"<style[^>]*>.*?</style>", b" ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(rb"<[^>]+>", b" ", text)
    text = re.sub(rb"&nbsp;", b" ", text, flags=re.IGNORECASE)
    text = re.sub(rb"&amp;", b"&", text, flags=re.IGNORECASE)
    text = re.sub(rb"\s+", b" ", text)
    return text.decode("utf-8", errors="replace").strip()


def build_universe_summary(db_path: str, ticker: str | None) -> str:
    if not ticker:
        return "issuer=None (not in current universe member list)"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM universe_members WHERE ticker = ? LIMIT 1", (ticker,)
    )
    row = cur.fetchone()
    if row is None:
        return f"issuer={ticker} (not in current universe member list)"
    return (
        f"issuer={row['ticker']} cik={row['cik']} exchange={row['exchange']} "
        f"market_cap={float(row['market_cap']):.0f} "
        f"prev_close={float(row['prev_close']):.2f}"
    )


def call_sonnet(api_request, api_key: str, *, max_tokens: int) -> tuple[str, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system_blocks = []
    for b in api_request.system:
        block = {"type": "text", "text": b.text}
        if b.cache_control:
            block["cache_control"] = b.cache_control
        system_blocks.append(block)
    messages = [
        {
            "role": m.role,
            "content": [{"type": "text", "text": b.text} for b in m.content],
        }
        for m in api_request.messages
    ]
    resp = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(
            resp.usage, "cache_creation_input_tokens", 0
        ),
    }
    return text, usage


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--filing-id", type=int, help="filings.id from the journal DB")
    g.add_argument("--accession", help="EDGAR accession number")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR), type=Path)
    p.add_argument(
        "--include-exhibits",
        action="store_true",
        help="Fetch and append Exhibit 99.x content (counterfactual mode)",
    )
    p.add_argument(
        "--exhibit-cap",
        type=int,
        default=DEFAULT_EXHIBIT_CAP_BYTES,
        help=f"Per-exhibit size cap in bytes (default {DEFAULT_EXHIBIT_CAP_BYTES})",
    )
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass the low-memory refusal (default: refuse if <250 MB available)",
    )
    args = p.parse_args()

    check_memory_or_exit(force=args.force)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not in environment. "
            "Try: set -a; . /etc/quinn/secrets.env; set +a"
        )

    filing = fetch_filing_row(
        args.db, filing_id=args.filing_id, accession=args.accession
    )
    print(f"=== FILING {filing.id} {filing.accession_number} ===")
    print(f"  form_type    = {filing.form_type}")
    print(f"  ticker       = {filing.issuer_ticker}")
    print(f"  item_codes   = {filing.item_codes}")
    print(f"  filed_at     = {filing.filed_at.isoformat()}")

    body_text = Path(filing.raw_text_path).read_text(
        encoding="utf-8", errors="replace"
    )
    print(f"  body length  = {len(body_text):,} chars")
    raw_text = body_text

    if args.include_exhibits:
        try:
            index = fetch_index(
                filing.cik, filing.accession_number, user_agent=args.user_agent
            )
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            raise SystemExit(f"failed to fetch EDGAR index: {e}")
        ex99_files = find_ex99_filenames(index)
        print(f"  ex-99 files  = {ex99_files or '(none found)'}")
        if ex99_files:
            parts = [body_text]
            for ex in ex99_files:
                try:
                    text = fetch_exhibit_text(
                        filing.cik,
                        filing.accession_number,
                        ex,
                        user_agent=args.user_agent,
                        cap=args.exhibit_cap,
                    )
                except (urllib.error.URLError, urllib.error.HTTPError) as e:
                    print(f"  ! {ex} fetch failed: {e}", file=sys.stderr)
                    continue
                parts.append(f"\n\n--- EXHIBIT {ex} ---\n\n")
                parts.append(text)
                print(f"  fetched {ex:30s} {len(text):>9,} chars")
            raw_text = "".join(parts)
            print(f"  augmented length = {len(raw_text):,} chars")

    builder = PromptBuilder(args.prompt_dir)
    decision_id = f"dryrun-{filing.id}-{'ex' if args.include_exhibits else 'noex'}"
    universe_summary = build_universe_summary(args.db, filing.issuer_ticker)
    ctx = AnalyzerContext(
        universe_summary=universe_summary,
        kill_switch_state="ok",
        open_positions_count=0,
        decision_id=decision_id,
    )
    request = builder.build_sonnet_filing_analysis(filing, raw_text, ctx)

    print()
    print(f"=== Calling Sonnet (prompt_version={request.prompt_version}) ===")
    print(f"  decision_id  = {decision_id}")
    print(f"  universe     = {universe_summary}")

    text, usage = call_sonnet(request, api_key, max_tokens=args.max_tokens)

    print()
    print(
        f"=== Response (in={usage['input_tokens']} out={usage['output_tokens']} "
        f"cache_read={usage['cache_read_input_tokens']}) ==="
    )

    try:
        parsed = json.loads(text)
        decision = parsed.get("decision") or (
            "trade_proposal" if "symbol" in parsed and "direction" in parsed else "?"
        )
        print(f"  decision     = {decision}")
        if "symbol" in parsed:
            print(f"  symbol       = {parsed.get('symbol')}")
            print(f"  direction    = {parsed.get('direction')}")
            print(f"  conviction   = {parsed.get('conviction')}")
            print(
                f"  size_pct     = {parsed.get('size_pct_of_capital')}"
            )
        thesis = parsed.get("thesis") or parsed.get("thesis_or_reason") or ""
        print()
        print("--- thesis / reason ---")
        print(thesis)
        if "signals_considered" in parsed:
            print()
            print("--- signals considered ---")
            for sig in parsed["signals_considered"]:
                print(f"  • {sig}")
    except json.JSONDecodeError:
        print("  [raw, not JSON-parseable]")
        print(text)


if __name__ == "__main__":
    main()
