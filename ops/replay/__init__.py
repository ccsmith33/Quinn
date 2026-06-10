"""Retrospective clean-execution replay tool (D-080, first half).

Standalone offline analysis: replays every journaled trade proposal under
honest simulated execution (next-open entry, GTC stop/TP per proposal
geometry, conservative touch ordering, time-stop) against historical daily
bars, and reports what the LLM's picks would have returned — uncontaminated
by the broken live exit layer.

Safety properties (hard constraints from task #5):
- Lives entirely under ``ops/replay/``; imports from ``src/`` are read-only
  and limited to ``config.calendar`` (NYSE trading-day predicate).
- Incapable of placing orders: no trading client is imported anywhere in
  this package; the only network egress is the Alpaca *market data* API
  (``data.alpaca.markets``), and only when not running ``--offline``.
- The journal database is opened in SQLite read-only mode.
"""
