# Quinn

Quinn is an autonomous LLM-driven swing-trading research system. It ingests SEC filings on a small/microcap universe, asks Sonnet (and, on high-conviction, Opus) to reason about each filing, sizes any resulting trade against tight risk caps, and submits orders through Alpaca paper. Architecture, decisions, and acceptance criteria are tracked under `artifacts/`; runtime ops live under `ops/`.

## Setup

To rehydrate Quinn on a fresh droplet, follow [`ops/runbooks/rehydrate.md`](ops/runbooks/rehydrate.md). The runbook is the proof of NFR-7 (full system rebuilt from git + B2 backup in ≤ 60 min) and is written for copy-paste execution.

For local development:

```bash
make install   # creates .venv, installs deps, applies migrations
make test      # smoke-check — full suite, ≤ 60s on a laptop
```

For product context — what Quinn is for, what it intentionally is not, and the numbered FR/NFRs the code must satisfy — see [`artifacts/planning/prd-quinn-v1.md`](artifacts/planning/prd-quinn-v1.md). The architecture doc is [`artifacts/design/architecture-quinn-v1.md`](artifacts/design/architecture-quinn-v1.md). Original Quinn v1 entry point details are in CLAUDE.md.

## Operator notes

### Ingestion cold-start: no RSS back-fill

On a fresh deploy, the RSS discovery loop (S3.2) does not back-fill historical filings — it seeds its seen-set from the first poll and only queues entries published after that moment (D-041). The submissions-API reconciler (S3.4) is the back-fill channel: it runs every 6 hours by default, diffing each in-universe issuer's recent filings (last 7 days) against the journal and queueing anything missed. After a fresh deploy, expect up to 6 hours before the first reconciler pass closes the cold-start gap; for an immediate top-up, call `Reconciler.force_reconcile_now()` from the agent main.

## Yearly maintenance

- **NYSE holiday calendar** at `src/config/nyse_holidays.json` (D-025) — refresh each December with the next two calendar years' NYSE-published full-day closures. Source: <https://www.nyse.com/markets/hours-calendars>. Half-day early closes are intentionally omitted (S3.2 RSS poll cadence treats them as full sessions).
