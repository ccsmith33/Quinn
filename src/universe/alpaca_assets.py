"""Alpaca `/v2/assets` fetcher.

Egress: `paper-api.alpaca.markets` or `api.alpaca.markets` (architecture §9.5).
The Alpaca SDK handles host selection from the configured endpoint.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class AlpacaAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    exchange: str
    status: str
    tradable: bool
    fractionable: bool
    asset_class: str


class _AlpacaClientLike(Protocol):
    def get_all_assets(self, status: str = "active") -> list[Any]: ...


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def fetch_alpaca_assets(client: _AlpacaClientLike) -> list[AlpacaAsset]:
    """Return active US-equity assets from Alpaca.

    Filters out crypto, ETFs marked under non-equity classes, etc.
    """
    raw = client.get_all_assets(status="active")
    out: list[AlpacaAsset] = []
    for a in raw:
        asset_class = _attr_or_key(a, "asset_class", "")
        if asset_class != "us_equity":
            continue
        out.append(
            AlpacaAsset(
                symbol=str(_attr_or_key(a, "symbol", "")),
                exchange=str(_attr_or_key(a, "exchange", "")),
                status=str(_attr_or_key(a, "status", "")),
                tradable=bool(_attr_or_key(a, "tradable", False)),
                fractionable=bool(_attr_or_key(a, "fractionable", False)),
                asset_class=str(asset_class),
            )
        )
    return out
