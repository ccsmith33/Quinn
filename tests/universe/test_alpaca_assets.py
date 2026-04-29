"""S2.1 — Alpaca /v2/assets fetcher."""

from __future__ import annotations

from dataclasses import dataclass

from universe.alpaca_assets import AlpacaAsset, fetch_alpaca_assets


@dataclass
class _FakeAsset:
    symbol: str
    exchange: str
    status: str
    tradable: bool
    fractionable: bool
    asset_class: str = "us_equity"


class _FakeAlpacaClient:
    def __init__(self, assets: list[_FakeAsset]) -> None:
        self._assets = assets
        self.calls: list[dict[str, str]] = []

    def get_all_assets(self, status: str = "active") -> list[_FakeAsset]:
        self.calls.append({"status": status})
        return self._assets


def test_returns_only_us_equity() -> None:
    client = _FakeAlpacaClient(
        [
            _FakeAsset("AAPL", "NASDAQ", "active", True, True, "us_equity"),
            _FakeAsset("BTCUSD", "FTXU", "active", True, True, "crypto"),
            _FakeAsset("ACME", "NYSE", "active", True, False, "us_equity"),
        ]
    )
    result = fetch_alpaca_assets(client)
    assert all(isinstance(a, AlpacaAsset) for a in result)
    symbols = {a.symbol for a in result}
    assert symbols == {"AAPL", "ACME"}


def test_calls_with_active_status() -> None:
    client = _FakeAlpacaClient([])
    fetch_alpaca_assets(client)
    assert client.calls == [{"status": "active"}]


def test_maps_fields() -> None:
    client = _FakeAlpacaClient(
        [_FakeAsset("AAPL", "NASDAQ", "active", True, True, "us_equity")]
    )
    [a] = fetch_alpaca_assets(client)
    assert a.symbol == "AAPL"
    assert a.exchange == "NASDAQ"
    assert a.status == "active"
    assert a.tradable is True
    assert a.fractionable is True
    assert a.asset_class == "us_equity"
