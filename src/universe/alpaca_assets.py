"""Alpaca `/v2/assets` fetcher.

Egress: `paper-api.alpaca.markets` or `api.alpaca.markets` (architecture §9.5).
The Alpaca SDK handles host selection from the configured endpoint.
"""

from __future__ import annotations

from typing import Any, Protocol

from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetAssetsRequest
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
    def get_all_assets(self, filter: Any | None = None) -> list[Any]: ...


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_str_value(v: Any) -> str:
    """Extract a bare string from an Alpaca enum or pass through a string.

    Alpaca SDK changed `Asset.exchange`, `Asset.status`, and `Asset.asset_class`
    from strings to enums; `enum_member.value` is the bare wire-format string
    (e.g. `AssetExchange.NYSE.value == "NYSE"`). Without this extraction,
    `str(enum_member)` yields `"AssetExchange.NYSE"` which fails downstream
    filters that compare against bare values.
    """
    return getattr(v, "value", str(v))


def fetch_alpaca_assets(client: _AlpacaClientLike) -> list[AlpacaAsset]:
    """Return active US-equity assets from Alpaca.

    Filters out crypto, ETFs marked under non-equity classes, etc.
    """
    raw = client.get_all_assets(filter=GetAssetsRequest(status=AssetStatus.ACTIVE))
    out: list[AlpacaAsset] = []
    for a in raw:
        asset_class = _extract_str_value(_attr_or_key(a, "asset_class", ""))
        if asset_class != "us_equity":
            continue
        out.append(
            AlpacaAsset(
                symbol=str(_attr_or_key(a, "symbol", "")),
                exchange=_extract_str_value(_attr_or_key(a, "exchange", "")),
                status=_extract_str_value(_attr_or_key(a, "status", "")),
                tradable=bool(_attr_or_key(a, "tradable", False)),
                fractionable=bool(_attr_or_key(a, "fractionable", False)),
                asset_class=asset_class,
            )
        )
    return out
