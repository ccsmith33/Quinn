# EDGAR fixtures

Per ADR-007 §"Test fixture standard", any test that exercises EDGAR
`index.json` parsing or primary-doc selection MUST use fixtures whose
`directory.item[]` entries match the shape of real EDGAR responses.

Anti-pattern (prohibited): fabricating an `index.json` with
`type: "<form-type-string>"` (e.g. `"8-K"`, `"10-K"`). Real EDGAR
returns icon-name strings like `text.gif`, `compressed.gif`,
`image2.gif`, `htm.gif`. Codifying form-type strings in a fixture is
how the original prose-form ingestion bug shipped.

## Files

| File | Source | Purpose |
| --- | --- | --- |
| `edgar_index_apple_10k.json` | `https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/index.json` (Apple 10-K, fiscal 2024) | Stage 1 heuristic: large 10-K with many exhibits, XBRL accessories, R\d+.htm fragments. Primary: `aapl-20240928.htm` (1.5 MB). |
| `edgar_index_apple_8k.json` | `https://www.sec.gov/Archives/edgar/data/320193/000032019326000011/index.json` (Apple 8-K, 2026-04-30) | Stage 1 heuristic: 8-K with primary smaller than its EX-99.1 earnings exhibit. Primary: `aapl-20260430.htm` (37 KB). |
| `edgar_index_altex_10q.json` | `https://www.sec.gov/Archives/edgar/data/775057/000109690626000672/index.json` (Altex Industries 10-Q, 2026-05-01) | Stage 1 heuristic: microcap 10-Q with a different filer-agent naming style (`altx-20260331_10q.htm`, `_ex` separator on exhibits). Primary: `altx-20260331_10q.htm` (184 KB). |
| `edgar_sgml_apple_8k_header.txt` | Header of `0000320193-26-000011.txt` (Apple 8-K above), bodies redacted | Stage 2 SGML fallback: real `<DOCUMENT>` block layout. Primary: `<TYPE>=8-K`, `<SEQUENCE>=1`, `<FILENAME>=aapl-20260430.htm`. |
| `edgar_sgml_altex_10q_header.txt` | Header of `0001096906-26-000672.txt` (Altex 10-Q above), bodies redacted | Stage 2 SGML fallback variant: different filer-agent block ordering. Primary: `<TYPE>=10-Q`, `<SEQUENCE>=1`, `<FILENAME>=altx-20260331_10q.htm`. |
| `edgar_sgml_coreweave_form4_header.txt` | Header of `0001104659-26-054312.txt` (CoreWeave Form 4, 2026-05-01), body redacted | Stage 2 SGML fallback: minimal single-`<DOCUMENT>` Form 4 layout. Primary: `<TYPE>=4`, `<SEQUENCE>=1`, `<FILENAME>=tm2613377-3_4seq1.xml`. Form 4 selection itself is unchanged per ADR-007 §"Form 4 path is unchanged"; this fixture supports cross-form SGML parser tests only. |

## Re-capturing fixtures

```bash
curl -A "Quinn-Stocks ccsmith33@crimson.ua.edu" \
  https://www.sec.gov/Archives/edgar/data/<cik>/<acc_no_dashes>/index.json \
  -o tests/ingestion/fixtures/<name>.json
```

SEC fair-access policy requires a declared User-Agent (FR-6 / NFR-17).
