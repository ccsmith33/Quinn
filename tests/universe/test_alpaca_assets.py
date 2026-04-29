"""S2.1 — Alpaca /v2/assets fetcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetAssetsRequest

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
        self.calls: list[Any] = []

    def get_all_assets(self, filter: Any | None = None) -> list[_FakeAsset]:
        self.calls.append(filter)
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


def test_calls_with_active_status_filter() -> None:
    client = _FakeAlpacaClient([])
    fetch_alpaca_assets(client)
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert isinstance(sent, GetAssetsRequest)
    assert sent.status == AssetStatus.ACTIVE


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


def test_extracts_enum_value_for_drifted_sdk_fields() -> None:
    """Alpaca SDK returns enums for `exchange`, `status`, and `asset_class`.

    Asset.exchange = AssetExchange.NYSE → str() yields "AssetExchange.NYSE"
    which fails the ALLOWED_EXCHANGES filter. The fetcher must extract
    `.value` (the bare wire-format string) so downstream filters match.
    """

    class _FakeEnum:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return f"FakeEnum.{self.value}"

    client = _FakeAlpacaClient(
        [
            _FakeAsset(
                symbol="AAPL",
                exchange=_FakeEnum("NYSE"),  # type: ignore[arg-type]
                status=_FakeEnum("active"),  # type: ignore[arg-type]
                tradable=True,
                fractionable=True,
                asset_class=_FakeEnum("us_equity"),  # type: ignore[arg-type]
            )
        ]
    )
    [a] = fetch_alpaca_assets(client)
    assert a.exchange == "NYSE"
    assert a.status == "active"
    assert a.asset_class == "us_equity"
