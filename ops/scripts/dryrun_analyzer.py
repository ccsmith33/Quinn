#!/usr/bin/env python3
"""Read-only dry-run of the analyzer pipeline against any model.

Fetches a filing's existing body from disk, optionally augments it with
its EDGAR Exhibit 99.x content (the press release / supplement that the
production ingest currently strips out for 8-K Item 2.02 / 7.01 / 8.01
filings), runs it through the production Sonnet prompt (byte-identical
via `prompts.loader.PromptBuilder`), and prints the decision + reasoning.

NO database writes. NO journal side effects. NO Opus reviewer call.
This is a manual diagnostic for the question "would Sonnet propose a
trade if it had the exhibit content?" — answer it without waiting for
the next market open.

Modes:

    # single filing, default model (Haiku 4.5)
    python ops/scripts/dryrun_analyzer.py --filing-id 487 --include-exhibits

    # single filing, override model
    python ops/scripts/dryrun_analyzer.py --filing-id 487 --include-exhibits \\
        --model claude-sonnet-4-6

    # batch over today's filings (cost-capped at ~\$3 unless --force)
    python ops/scripts/dryrun_analyzer.py --batch-today --include-exhibits

    # batch over the last two days
    python ops/scripts/dryrun_analyzer.py --batch-today --since 2026-05-05

The script enforces a 250 MB memory floor (refuses on tight hosts) and a
\$3 batch-cost ceiling (override with `--force`). On the production
droplet (2 GB RAM post-resize) both gates pass for typical batches.

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
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
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


def call_anthropic(
    api_request, api_key: str, *, model: str, max_tokens: int
) -> tuple[str, dict]:
    """Direct Anthropic SDK call against any model id (Sonnet, Haiku, Opus)."""
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
        model=model,
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


# Public per-1M-token rates so the script can self-report estimated cost
# without depending on the project's pricing module. Values mirror
# `src/analyzer/pricing.py` as of 2026-05-06; if rates change there,
# update here too (or import — but keeping this script self-contained
# is more robust against import breakage).
_RATES_PER_MILLION = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}


def estimate_cost(model: str, usage: dict) -> float:
    rates = _RATES_PER_MILLION.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    plain_input = usage["input_tokens"] - cache_read - cache_create
    return (
        plain_input * in_rate / 1_000_000
        + cache_read * (in_rate * 0.1) / 1_000_000
        + cache_create * (in_rate * 1.25) / 1_000_000
        + usage["output_tokens"] * out_rate / 1_000_000
    )


def fetch_today_filing_ids(db_path: str, *, since: str | None = None) -> list[int]:
    """Return all filing.id values from the journal where filed_at falls in
    `since` (default: today). Used by --batch-today.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if since is None:
        since = "date('now')"
        cur = conn.execute(
            f"SELECT id FROM filings WHERE filed_at >= {since} ORDER BY id"
        )
    else:
        cur = conn.execute(
            "SELECT id FROM filings WHERE filed_at >= ? ORDER BY id", (since,)
        )
    return [row["id"] for row in cur.fetchall()]


def analyze_one(
    *,
    filing: FilingRow,
    db_path: str,
    api_key: str,
    model: str,
    prompt_dir: Path,
    include_exhibits: bool,
    exhibit_cap: int,
    user_agent: str,
    max_tokens: int,
    verbose: bool,
) -> dict:
    """Run a single dryrun analysis and return a result dict.

    `verbose=True` prints the per-filing detail block (used by single-filing
    mode); `verbose=False` prints only one line per filing (used by batch).
    """
    body_text = Path(filing.raw_text_path).read_text(
        encoding="utf-8", errors="replace"
    )
    raw_text = body_text
    ex99_files: list[str] = []

    if include_exhibits:
        try:
            index = fetch_index(
                filing.cik, filing.accession_number, user_agent=user_agent
            )
            ex99_files = find_ex99_filenames(index)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  ! index fetch failed for {filing.id}: {e}", file=sys.stderr)
            ex99_files = []
        if ex99_files:
            parts = [body_text]
            for ex in ex99_files:
                try:
                    text = fetch_exhibit_text(
                        filing.cik,
                        filing.accession_number,
                        ex,
                        user_agent=user_agent,
                        cap=exhibit_cap,
                    )
                except (urllib.error.URLError, urllib.error.HTTPError) as e:
                    print(f"  ! {ex} fetch failed: {e}", file=sys.stderr)
                    continue
                parts.append(f"\n\n--- EXHIBIT {ex} ---\n\n")
                parts.append(text)
            raw_text = "".join(parts)

    if verbose:
        print(f"=== FILING {filing.id} {filing.accession_number} ===")
        print(f"  form_type    = {filing.form_type}")
        print(f"  ticker       = {filing.issuer_ticker}")
        print(f"  item_codes   = {filing.item_codes}")
        print(f"  filed_at     = {filing.filed_at.isoformat()}")
        print(f"  body length  = {len(body_text):,} chars")
        if include_exhibits:
            print(f"  ex-99 files  = {ex99_files or '(none found)'}")
            if ex99_files:
                print(f"  augmented length = {len(raw_text):,} chars")

    builder = PromptBuilder(prompt_dir)
    decision_id = f"dryrun-{filing.id}-{'ex' if include_exhibits else 'noex'}"
    universe_summary = build_universe_summary(db_path, filing.issuer_ticker)
    ctx = AnalyzerContext(
        universe_summary=universe_summary,
        kill_switch_state="ok",
        open_positions_count=0,
        decision_id=decision_id,
    )
    request = builder.build_sonnet_filing_analysis(filing, raw_text, ctx)

    if verbose:
        print()
        print(f"=== Calling {model} (prompt_version={request.prompt_version}) ===")
        print(f"  decision_id  = {decision_id}")
        print(f"  universe     = {universe_summary}")

    text, usage = call_anthropic(
        request, api_key, model=model, max_tokens=max_tokens
    )
    cost = estimate_cost(model, usage)

    parsed: dict | None = None
    decision = "?"
    conviction: int | None = None
    symbol: str | None = None
    thesis = ""
    try:
        parsed = json.loads(text)
        decision = parsed.get("decision") or (
            "trade_proposal"
            if "symbol" in parsed and "direction" in parsed
            else "?"
        )
        conviction = parsed.get("conviction")
        symbol = parsed.get("symbol")
        thesis = parsed.get("thesis") or parsed.get("thesis_or_reason") or ""
    except json.JSONDecodeError:
        pass

    if verbose:
        print()
        print(
            f"=== Response (in={usage['input_tokens']} out={usage['output_tokens']} "
            f"cache_read={usage['cache_read_input_tokens']}) ==="
        )
        print(f"  decision     = {decision}")
        if symbol:
            print(f"  symbol       = {symbol}")
            print(f"  direction    = {parsed.get('direction') if parsed else None}")
            print(f"  conviction   = {conviction}")
            print(f"  size_pct     = {parsed.get('size_pct_of_capital') if parsed else None}")
        if parsed is not None:
            print()
            print("--- thesis / reason ---")
            print(thesis)
            if "signals_considered" in parsed:
                print()
                print("--- signals considered ---")
                for sig in parsed["signals_considered"]:
                    print(f"  • {sig}")
        else:
            print("  [raw, not JSON-parseable]")
            print(text)

    return {
        "filing_id": filing.id,
        "ticker": filing.issuer_ticker,
        "form_type": filing.form_type,
        "item_codes": filing.item_codes,
        "decision": decision,
        "conviction": conviction,
        "symbol": symbol,
        "thesis_head": thesis[:120],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cache_read": usage["cache_read_input_tokens"],
        "cost_usd": cost,
        "parse_ok": parsed is not None,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--filing-id", type=int, help="filings.id from the journal DB")
    g.add_argument("--accession", help="EDGAR accession number")
    g.add_argument(
        "--batch-today",
        action="store_true",
        help="Run dryrun on every filing with filed_at >= today (cost: ~$0.03/call x N filings on Haiku; flagged if >$3)",
    )
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR), type=Path)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model id (default {DEFAULT_MODEL}). "
        f"Known: {', '.join(_RATES_PER_MILLION.keys())}",
    )
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
        "--limit",
        type=int,
        default=200,
        help="Max filings to process in --batch-today (safety cap, default 200)",
    )
    p.add_argument(
        "--since",
        default=None,
        help="ISO date for --batch-today window (default: today). "
        "E.g. '2026-05-05' to include yesterday too.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass the low-memory refusal AND the >$3 cost confirmation",
    )
    args = p.parse_args()

    check_memory_or_exit(force=args.force)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not in environment. "
            "Try: set -a; . /etc/quinn/secrets.env; set +a"
        )

    if args.model not in _RATES_PER_MILLION:
        print(
            f"WARNING: unknown model '{args.model}' — cost estimates will be 0. "
            f"Known: {', '.join(_RATES_PER_MILLION.keys())}",
            file=sys.stderr,
        )

    if args.batch_today:
        run_batch(args, api_key)
    else:
        filing = fetch_filing_row(
            args.db, filing_id=args.filing_id, accession=args.accession
        )
        analyze_one(
            filing=filing,
            db_path=args.db,
            api_key=api_key,
            model=args.model,
            prompt_dir=args.prompt_dir,
            include_exhibits=args.include_exhibits,
            exhibit_cap=args.exhibit_cap,
            user_agent=args.user_agent,
            max_tokens=args.max_tokens,
            verbose=True,
        )


def run_batch(args, api_key: str) -> None:
    """Iterate over today's filings (or `--since`'s window), run one analysis
    each, print a one-line summary per filing, then aggregate stats at end.
    """
    filing_ids = fetch_today_filing_ids(args.db, since=args.since)
    if args.limit and len(filing_ids) > args.limit:
        print(
            f"NOTE: {len(filing_ids)} filings match window; capping at --limit {args.limit}.",
            file=sys.stderr,
        )
        filing_ids = filing_ids[: args.limit]

    rates = _RATES_PER_MILLION.get(args.model, (3.0, 15.0))
    est_per_call = (rates[0] * 25_000 + rates[1] * 700) / 1_000_000
    est_total = est_per_call * len(filing_ids)
    print(
        f"=== BATCH DRYRUN ===\n"
        f"  model            = {args.model}\n"
        f"  filings          = {len(filing_ids)}\n"
        f"  include_exhibits = {args.include_exhibits}\n"
        f"  est. cost        = ~${est_total:.2f} "
        f"(~${est_per_call:.4f}/call × {len(filing_ids)})\n"
    )
    if est_total > 3.0 and not args.force:
        raise SystemExit(
            f"REFUSED: estimated batch cost ~${est_total:.2f} exceeds $3 safety cap. "
            f"Pass --force to proceed."
        )

    results: list[dict] = []
    print(
        f"{'id':>5} {'ticker':<8} {'form':<8} {'decision':<14} "
        f"{'conv':>4} {'in_tok':>6} {'cost':>7}  {'thesis_head':<60}"
    )
    print("-" * 120)
    for i, fid in enumerate(filing_ids, 1):
        try:
            filing = fetch_filing_row(
                args.db, filing_id=fid, accession=None
            )
        except SystemExit:
            print(f"  ! filing {fid} not found, skipping", file=sys.stderr)
            continue
        try:
            r = analyze_one(
                filing=filing,
                db_path=args.db,
                api_key=api_key,
                model=args.model,
                prompt_dir=args.prompt_dir,
                include_exhibits=args.include_exhibits,
                exhibit_cap=args.exhibit_cap,
                user_agent=args.user_agent,
                max_tokens=args.max_tokens,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001 — diagnostic script; want to keep going
            print(
                f"  ! filing {fid} ({filing.issuer_ticker}) raised: {e}",
                file=sys.stderr,
            )
            continue
        results.append(r)
        print(
            f"{r['filing_id']:>5} {(r['ticker'] or '-'):<8} "
            f"{filing.form_type:<8} {r['decision']:<14} "
            f"{(r['conviction'] if r['conviction'] is not None else '-'):>4} "
            f"{r['input_tokens']:>6} ${r['cost_usd']:>6.4f}  {r['thesis_head']:<60}"
        )

    # ---- summary ----
    print()
    print("=== SUMMARY ===")
    n = len(results)
    if n == 0:
        print("  no successful analyses.")
        return
    decisions: dict[str, int] = {}
    convictions: dict[int, int] = {}
    total_cost = 0.0
    total_in = 0
    total_out = 0
    parse_failures = 0
    for r in results:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
        if r["conviction"] is not None:
            convictions[r["conviction"]] = convictions.get(r["conviction"], 0) + 1
        total_cost += r["cost_usd"]
        total_in += r["input_tokens"]
        total_out += r["output_tokens"]
        if not r["parse_ok"]:
            parse_failures += 1
    print(f"  analyzed         = {n}")
    print(f"  total cost (est) = ${total_cost:.2f}")
    print(f"  avg per call     = ${total_cost / n:.4f}")
    print(f"  total tokens     = {total_in:,} in / {total_out:,} out")
    print(f"  parse failures   = {parse_failures}")
    print()
    print("  decisions:")
    for k, v in sorted(decisions.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20} {v}")
    if convictions:
        print()
        print("  conviction (where present):")
        for k in sorted(convictions):
            print(f"    {k:<2} {convictions[k]}")
    trade_proposals = [r for r in results if r["decision"] == "trade_proposal"]
    if trade_proposals:
        print()
        print(f"  TRADE PROPOSALS ({len(trade_proposals)}):")
        for r in trade_proposals:
            print(
                f"    {r['ticker'] or '-':<8} conviction={r['conviction']} "
                f"thesis={r['thesis_head']}"
            )


if __name__ == "__main__":
    main()
