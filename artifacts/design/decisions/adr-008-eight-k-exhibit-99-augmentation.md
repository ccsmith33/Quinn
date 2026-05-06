---
id: ADR-008
title: 8-K Exhibit 99.x augmentation for furnish-item filings
status: accepted
date: 2026-05-05
phase: implementation
authors: dev-ingestion
relates_to: ADR-002, ADR-007, FR-1, FR-2, FR-5, FR-6
extends: ADR-007 (additive post-primary augmentation; ADR-007's primary-doc selection is unchanged)
---

# ADR-008 — 8-K Exhibit 99.x augmentation for furnish-item filings

## Status

Accepted. Companion to ADR-007 (does NOT replace it). ADR-007's two-stage primary-doc selection still owns body selection for every prose form. This ADR adds an additive, post-primary exhibit-fetch step that fires only for the narrow class of 8-Ks whose substance is in Exhibit 99.x.

## Context

Production observation 2026-05-05: 58/58 of the day's evaluated proposals landed `no_trade`, including ~10 earnings releases (SWIM, HCKT, OUST, etc., all 8-K Item 2.02). Sonnet's recurring decline reason: *"the actual financial data is in the attached Exhibit 99, which is not reproduced in the filing body — no revenue, EPS, or guidance figures are available for analysis."*

This is the second consecutive day of zero trades. Yesterday's bug (ADR-007 + the 2026-05-04 ticker-resolver hotfix) shipped successfully — every today's filing now has a populated `issuer_ticker` and the prose primary-doc body is correctly selected. But the analyzer is still being fed a metadata wrapper, not the press release.

### Root cause

8-K filings using these item codes furnish — not file — their substance via attached Exhibit 99.x:

- **Item 2.02 — Results of Operations and Financial Condition.** The body is a one-page cover sheet referencing "the press release attached as Exhibit 99.1, which is hereby furnished and not filed." The actual revenue/EPS/guidance numbers are in `ex-99-1.htm`.
- **Item 7.01 — Regulation FD Disclosure.** Body cites the attached presentation deck or transcript; the substance is in EX-99.x.
- **Item 8.01 — Other Events.** Catch-all for material events (strategic partnerships, lawsuits, dividend declarations); body is typically a one-line pointer to EX-99.

ADR-007's primary-doc selection (`detail_fetcher._select_prose_primary`) explicitly excludes EX-99 via the regex `(^|[-_])ex(hibit)?[-_]?\d` (line 67). For 8-K Items 1.01/3.01/5.02/etc. (where the body IS the substance), this is correct — including the EX-99 there would dilute the analyzer's context. But for the three furnish-item codes above, excluding the exhibit means ingesting only the cover sheet.

### Why the prior fix didn't catch this

ADR-007 was scoped to fix prose-form *primary-doc* selection (the system was ingesting 0% of 8-Ks before that fix). The exhibit-exclusion regex is appropriate for primary-doc selection — exhibits are NOT the primary doc. But the implicit assumption was that the primary doc carries the substance. For Items 2.02/7.01/8.01, that assumption is false.

The 2026-05-04 ticker-resolver hotfix unblocked the universe gate and let these filings through to Sonnet. Today is the first day Sonnet has actually seen an 8-K Item 2.02 and refused it for lack of figures. Pre-fix, the universe gate was rejecting them silently.

## Decision

**Adopt Option A: additively append every EX-99.x exhibit's content to `raw_text` for 8-Ks whose `item_codes` intersect `{2.02, 7.01, 8.01}`, walking exhibits in numeric order under a cumulative byte cap.**

The augmentation is additive (no schema change, no analyzer prompt change), is fired only for the narrow furnish-item subset (no impact on Items 1.01/3.01/5.02/etc.), and degrades gracefully on every failure mode (missing exhibit, fetch error, cumulative-cap overshoot, empty plaintext → partial-or-zero exhibit set with a logged warning, never a loop crash).

### Algorithm

After ADR-007 Stage 1+2 selects the primary doc and `_extract_item_codes` parses the body for `Item N.NN` headings:

1. If `form_type == "8-K"` AND any extracted item code is in `_FURNISH_ITEM_CODES = frozenset({"2.02", "7.01", "8.01"})`:
2. Scan `directory.item[]` for ALL EX-99.x `.htm`/`.html` files (regex `(?:^|[-_a-z0-9]*?)ex[-_.]?99[-_.]?(\d+)[-_a-z0-9]*?\.html?$`, capturing the numeric suffix). The pattern accepts canonical SEC shapes (`ex-99-1.htm`, `EX-99.1.HTM`), filer-agent shapes (`a8-kex991.htm` from Apple, `d8-kex991.htm` from Donnelley, `tm2613377-3_4ex99-1.htm`), and trailing-token suffixes (`a8-kex991q2202603282026.htm`). Sort ascending by exhibit number. **Known limitation**: shapes that omit the literal `ex` token (e.g., `tm26-3-99-1.htm`) are not matched — broadening would false-positive on date-like filenames; SGML-header fallback (per ADR-007 Stage 2 mechanism) is a tracked follow-up if production surfaces these.
3. Walk the sorted list; for each exhibit:
   - Fetch via `EdgarClient.get_bounded(url, max_bytes=remaining_budget)` (the shared client, rate-limited per ADR-002 §6 — never instantiate a parallel client; concurrent calls share the same rate limiter and circuit breaker). `get_bounded` **streams** the response and aborts the connection the moment the per-fetch budget (= cumulative cap minus bytes accumulated so far) is exceeded. This is the only line of defense against an adversarial filing attaching a 500 MB EX-99 — `httpx.AsyncClient.get()` would buffer the full body into memory before any post-fetch check could fire.
   - On `EdgarUnavailable` / non-200 status / empty plaintext: log warning, **skip to the next exhibit** (a single bad fetch never aborts the rest of the loop).
   - On `BoundedResponse(truncated=True)` (server tried to send more than the remaining budget): log warning, **stop the loop** (later exhibits would only push us further over).
   - Else: append `--- EXHIBIT 99.{n} ---` separator + `html_to_text(body)` to a running list of chunks.
4. If at least one exhibit's text was appended, write `body + "".join(chunks)` to `raw_text_path`; else write body-only.
5. Filing always lands as `ingest_state="ok"` — degraded ingest (zero or partial exhibit set) matches pre-fix behavior, and the loop never crashes on this path.

The `_FURNISH_ITEM_CODES` constant is the single edit point for adding new codes (e.g., 5.02 if officer-change bios start landing exclusively in EX-99). Per the reviewer's checklist, the constant lives at the top of `detail_fetcher.py` next to other module-level configuration.

### Multiple-exhibit decision

Earnings 8-Ks frequently attach 99.1 (press release), 99.2 (financial supplement / non-GAAP reconciliations), and 99.3 (slide deck). The chosen behavior is **"fetch ALL EX-99.x exhibits in numeric order, with a cumulative byte cap across all of them."**

Rationale:

- **Each exhibit carries independent signal.** 99.1 is the headline narrative + GAAP figures + guidance; 99.2 contains non-GAAP reconciliations the analyst may rely on for a more accurate forward read; 99.3 (slide deck) often includes additional segment-level commentary not in the press release. Excluding 99.2/99.3 leaves analyst signal on the table.
- **Cumulative cap protects the loop against OOM (streaming-enforced).** A 5 MB hard ceiling on the **combined** body bytes (across all fetched exhibits) means a pathologically large filing cannot OOM the agent. Enforcement point matters: the cap is enforced **mid-stream** via `EdgarClient.get_bounded`, not after a buffered fetch. `httpx.AsyncClient.get()` would buffer a 500 MB exhibit into RAM before any application code could check its size — `get_bounded` instead uses `httpx.AsyncClient.stream(...)` + `aiter_bytes()` and aborts the connection the moment cumulative bytes exceed the per-fetch budget. The cap is the only line of defense against an adversarial or accidentally-huge response (this fix addressed reviewer finding B-2).
- **Numeric ordering is predictable.** Walking 99.1 → 99.2 → 99.3 keeps the raw_text layout consistent across filings (analyst sees the press release first, then supplement, then deck) and makes ops debugging deterministic.
- **Bandwidth bound.** Typical earnings 8-K combined exhibit set is 50 KB – 2 MB; the 5 MB cap leaves 1–2x headroom for the median case while still aborting on the genuine outliers.
- **Per-exhibit failures don't cascade.** If 99.1 returns a 503 but 99.2 + 99.3 succeed, the latter still get appended. A single bad fetch never blocks the rest.

Considered alternatives:

- **Fetch only the lowest-numbered (99.1).** Rejected — leaves 99.2's reconciliations and 99.3's segment commentary out of the analyst's context. Initial draft of this ADR proposed this; reviewer feedback flagged it as too conservative given that earnings PRs commonly span both the press release AND the supplement. Easy to reverse to multi-fetch later, but reverting from multi-fetch back to single requires re-justifying the cut.
- **Concatenate all 99.x with a per-file cap (e.g., 5 MB each).** Rejected — three 5 MB exhibits = 15 MB raw_text, way past what the analyzer prompt budget can absorb. The cap must be cumulative.
- **Fetch the largest 99.x by size.** Rejected — non-deterministic; supplements are sometimes larger than the headline press release but are the wrong content for headline narrative.
- **Truncate an oversized exhibit instead of skipping it.** Rejected — partial-document plaintext is misleading (a truncated press release loses the bottom-line numbers). Better to skip and degrade cleanly.

### Failure modes

Every degraded path keeps the filing as `ingest_state="ok"`. Worst case: zero exhibits appended, `raw_text` is body-only (matches pre-fix behavior). The agent loop must never crash on the exhibit path:

| Failure mode                                          | Behavior                                          | Log event                                  |
|-------------------------------------------------------|---------------------------------------------------|--------------------------------------------|
| No EX-99 in `directory.item[]`                        | Body-only; no log                                 | (silent — common case for old filings)     |
| One exhibit fetch returns 4xx/5xx                     | Skip that exhibit; continue with the rest        | `exhibit_fetch_non_200`                    |
| One exhibit fetch raises `EdgarUnavailable`           | Skip that exhibit; continue with the rest        | `exhibit_fetch_unavailable`                |
| Adding next exhibit would exceed cumulative cap (5 MB)| Stop loop; keep what's already appended           | `exhibit_cumulative_cap_reached`           |
| `html_to_text` returns empty for one exhibit          | Skip that exhibit; continue with the rest         | (silent — rare; degenerate HTML)           |
| 8-K item code outside furnish set                     | Body-only; no fetch attempted                     | (silent — by design)                       |
| `form_type != "8-K"`                                  | Body-only; no fetch attempted                     | (silent — by design)                       |

### Why not Option B (replace primary doc with EX-99)

Considered: for furnish-item 8-Ks, swap the primary-doc pick from the body to EX-99.1. Rejected because:

1. The 8-K body still carries useful metadata: signing officer, exact filed-at timestamp, item-code list confirmation, the "furnished not filed" Reg FD disclaimer (informative for the analyst's assessment of disclosure quality). Dropping the body throws this away.
2. `extract_item_codes` runs against the body; reliable item-code extraction would need a separate pre-fetch pass. Additive append keeps the pipeline linear.
3. Reverting if EX-99 fetch fails would leave the body as primary anyway — same code path as Option A, just more branching.

Option A is the conservative, additive, easily-reversible path.

### Why not Option C (separate `exhibit_text_path` column)

Considered: schema-migrate to add `filings.exhibit_text_path TEXT`, write the exhibit to a separate file, plumb a second path into the analyzer's prompt builder. Rejected:

1. Schema migration touches cross-cutting code (`journal/migrate.py`, `models.py`, `repo.py`, every code path that reads `FilingRow`).
2. Analyzer prompt change touches the prompt registry (ADR-005 prompt versioning), which would require new `prompt_version` and re-baselining of cost telemetry.
3. The hotfix constraint was "no analyzer prompt changes, no schema migrations." Option A respects both.

If a future v2 needs separate exhibit text (e.g., for distinct similarity-prefilter handling), the migration is a clean additive change at that point.

## Consequences

### Positive

- 8-K Items 2.02/7.01/8.01 filings finally land with the substantive content Sonnet needs to make a trade decision. Today's SWIM, HCKT, OUST, and similar earnings filings should re-analyze with revenue/EPS/guidance figures in context, plus segment commentary from the supplement and slide deck where attached.
- No schema change, no prompt change, no new dependencies. Surgical and reversible.
- Graceful-degradation discipline is preserved: every exhibit failure mode keeps the filing as `ingest_state="ok"`. Worst case is body-only `raw_text`, identical to pre-fix behavior. A single bad fetch never aborts the loop, and the cumulative byte cap protects against OOM.
- The `_FURNISH_ITEM_CODES` constant and `_select_all_ex_99` helper are the single extension points. Adding a new code (e.g., 5.02 for officer-change bios) is a one-line edit; tightening the EX-99 regex is a localized change.

### Negative

- Up to N additional rate-limited EDGAR fetches per qualifying 8-K (one per attached EX-99 exhibit, typically 1–3). Bounded: only filings with item codes in the furnish set, exhibits stop accumulating at the 5 MB cumulative body cap, all fetches share the global 10 req/sec rate limiter (ADR-002 §6). Real-world rate is ~30–40% of 8-Ks.
- `raw_text` for an earnings filing now spans body + 1–N EX-99.x, typically 50 KB – 2 MB up from 5–20 KB. Downstream prompt-caching may need re-tuning if cache hit rates drop materially. Acceptable cost for the analyzer correctness gain.
- The cumulative cap means the very last exhibits in a multi-exhibit filing may be silently skipped if earlier ones already filled the budget. Tracked via the `exhibit_cumulative_cap_reached` log event; if production shows this firing on legitimate filings, raise the cap.

### Neutral

- Observability: the `exhibit_fetch_non_200` / `exhibit_fetch_unavailable` / `exhibit_cumulative_cap_reached` warnings are structured logs the operator can monitor. A Prometheus metric (`edgar_exhibit_fetch_count{result="ok|missing|http_error|cap_reached|unavailable"}`) is a natural follow-up but out of scope here.
- The existing exhibit-exclusion regex in `_select_prose_primary` is unchanged — exhibits are still NOT primary-doc candidates. ADR-008 lives entirely in the post-primary augmentation path.
- Concurrency: RSS loop and `Reconciler` both feed `DetailFetcher`. The new exhibit fetches go through the same shared `EdgarClient` (via `self._edgar.get(...)`), so the global rate limiter and circuit breaker apply uniformly. The `DuplicateAccession` guard in `insert_filing` still owns idempotency; the exhibit step happens before persistence and never creates a parallel ingest path.

## Test scenarios (mandatory)

These map 1:1 to the test file `tests/ingestion/test_detail_fetcher.py` additions:

- **Happy path**: 8-K with Item 2.02 + `ex-99-1.htm` → `raw_text` contains both body's `Item 2.02` heading and the exhibit's distinctive numerical content (revenue, EPS).
- **Each furnish item triggers**: separate tests for 2.02, 7.01, 8.01 — each must fetch EX-99.x.
- **Multi-item with one furnish code**: 8-K with items `{2.02, 5.07}` → exhibits ARE fetched (any furnish-item code suffices).
- **Non-furnish item only**: 8-K with item 5.02 + an EX-99 in the index → exhibit NOT fetched (regression guard against bloating officer-change filings).
- **No EX-99 in index**: furnish item but the filer didn't attach an exhibit → body-only ingest, no crash.
- **EX-99 fetch 5xx (single exhibit)**: body-only ingest, no crash, warning logged.
- **10-Q with EX-99 in index**: NOT fetched (only 8-Ks trigger).
- **Multiple EX-99 exhibits (99.1 + 99.2 + 99.3)**: ALL three appended in numeric order; per-exhibit separators present; ordering is deterministic regardless of `index.json` order.
- **Cumulative byte cap stops further exhibits**: 99.1 fits but 99.1+99.2 overshoots → 99.1 in raw_text, 99.2 and 99.3 not.
- **First-exhibit-alone-overshoots → body-only**: 6 MB single exhibit > 5 MB cap → ingest still OK, no augmentation.
- **One exhibit 5xx does not block others**: 99.1 returns 503, 99.2 returns 200 → 99.2 still appended; 99.1 absent.
- **Separator marks the boundary**: the `--- EXHIBIT 99.{n} ---` separator appears between body and each exhibit.
- **Filename-variant coverage**: `ex-99-1.htm`, `ex99-1.htm`, `ex991.htm`, `ex_99_1.htm` all match.
- **Non-99 exhibit (`ex-31-1.htm`) not fetched**: officer cert exhibits are out of scope.
- **`_select_all_ex_99` unit tests**: ascending-sort selection, empty list when no 99, variant matching, non-HTML extension exclusion.
- **`_FURNISH_ITEM_CODES` constant test**: locks the set to `{"2.02", "7.01", "8.01"}` so a future grep can find it.

## Cross-references

- **Companion to**: ADR-007 (prose primary-doc selection — unchanged for body pick; this ADR augments after).
- **Relates to**: ADR-002 §6 (rate-limit posture — exhibit fetch consumes one slot per qualifying 8-K), ADR-005 (no prompt-version change required — additive `raw_text` only).
- **Implementation owner**: dev-ingestion (task #1).
- **Backfill owner**: dev-ingestion (task #2 — `ops/scripts/backfill_8k_exhibits.py` rescues Mon+Tue 8-Ks ingested without exhibit content).
- **Review owner**: reviewer (task #3).
