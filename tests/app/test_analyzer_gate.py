"""Pre-analysis buyability gate — pure-logic tests (API cost lever).

The load-bearing property is FAIL OPEN: only an affirmative, well-formed
"this filer cannot be bought" answer may skip the LLM call. Every
ambiguity analyzes anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analyzer_gate import (
    Analyze,
    FailOpen,
    SkipUnbuyable,
    evaluate_buyability,
)

_CIK = 320193


@dataclass(frozen=True)
class _Member:
    ticker: str
    prev_close: float


class _FakeUniverse:
    def __init__(
        self, members: dict[str, float], *, ciks: set[int] | None = None
    ) -> None:
        self._members = {t: _Member(t, p) for t, p in members.items()}
        self._ciks = set() if ciks is None else ciks

    def get_member(self, ticker: str) -> _Member | None:
        return self._members.get(ticker)

    def is_in_universe_by_cik(self, cik: int) -> bool:
        return cik in self._ciks

    def member_count(self) -> int:
        return len(self._members)


def _decide(
    *,
    ticker: str | None,
    universe: object,
    price_floor_usd: float = 5.0,
    cik: int | None = _CIK,
) -> object:
    return evaluate_buyability(
        ticker=ticker,
        cik=cik,
        universe=universe,  # type: ignore[arg-type]
        price_floor_usd=price_floor_usd,
    )


def test_in_universe_and_in_band_analyzes() -> None:
    decision = _decide(ticker="ACME", universe=_FakeUniverse({"ACME": 100.0}))
    assert isinstance(decision, Analyze)


def test_at_the_floor_exactly_analyzes() -> None:
    """The validator rejects strictly BELOW the floor; the gate must not
    be stricter than the check it is anticipating."""
    decision = _decide(ticker="ACME", universe=_FakeUniverse({"ACME": 5.0}))
    assert isinstance(decision, Analyze)


def test_out_of_universe_skips() -> None:
    decision = _decide(ticker="NOPE", universe=_FakeUniverse({"ACME": 100.0}))
    assert isinstance(decision, SkipUnbuyable)
    assert decision.reason == "universe"
    assert decision.symbol == "NOPE"
    assert decision.price is None


def test_ticker_miss_with_in_universe_cik_fails_open() -> None:
    """Multi-class listings and a stale ticker cache make the resolver's
    ticker disagree with the snapshot. The prefilter already accepted this
    CIK, so the disagreement is a resolver defect — not an answer."""
    decision = _decide(
        ticker="GOOG",
        universe=_FakeUniverse({"GOOGL": 150.0}, ciks={_CIK}),
    )
    assert isinstance(decision, FailOpen)
    assert decision.detail == "ticker_snapshot_mismatch"


def test_below_price_floor_skips_and_reports_the_price() -> None:
    decision = _decide(ticker="PENNY", universe=_FakeUniverse({"PENNY": 3.25}))
    assert isinstance(decision, SkipUnbuyable)
    assert decision.reason == "price"
    assert decision.price == 3.25


def test_configured_floor_is_honored_not_the_default() -> None:
    """The gate reuses the validator's configured floor, so raising it
    moves the gate with it. This is the case that actually bites in
    production — the snapshot's own screen is the same $5 default."""
    universe = _FakeUniverse({"MID": 6.0})
    assert isinstance(_decide(ticker="MID", universe=universe), Analyze)
    decision = _decide(ticker="MID", universe=universe, price_floor_usd=10.0)
    assert isinstance(decision, SkipUnbuyable)
    assert decision.reason == "price"


def test_unresolvable_ticker_fails_open() -> None:
    for ticker in (None, ""):
        decision = _decide(ticker=ticker, universe=_FakeUniverse({"ACME": 100.0}))
        assert isinstance(decision, FailOpen), ticker
        assert decision.detail == "unresolved_ticker"


def test_empty_universe_snapshot_fails_open() -> None:
    """An empty snapshot answers "not a member" to every ticker — it must
    never be read as a universe-wide rejection."""
    decision = _decide(ticker="ACME", universe=_FakeUniverse({}))
    assert isinstance(decision, FailOpen)
    assert decision.detail == "empty_universe_snapshot"


def test_universe_error_fails_open() -> None:
    class _ExplodingUniverse:
        def get_member(self, ticker: str) -> object | None:
            raise RuntimeError("snapshot table vanished")

        def is_in_universe_by_cik(self, cik: int) -> bool:
            return False

        def member_count(self) -> int:
            return 3

    decision = _decide(ticker="ACME", universe=_ExplodingUniverse())
    assert isinstance(decision, FailOpen)
    assert decision.detail == "universe_error:RuntimeError"


def test_member_without_a_price_fails_open() -> None:
    class _PricelessUniverse:
        def get_member(self, ticker: str) -> object:
            return object()

        def is_in_universe_by_cik(self, cik: int) -> bool:
            return True

        def member_count(self) -> int:
            return 1

    decision = _decide(ticker="ACME", universe=_PricelessUniverse())
    assert isinstance(decision, FailOpen)
    assert decision.detail == "no_snapshot_price"
