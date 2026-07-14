"""Alpaca news polling — the `event:news` trigger's data source.

Minimal wrapper over alpaca-py 0.43.x's `NewsClient` (REST
`/v1beta1/news` behind the scenes; available on the free data plan with
the same credential pair the trading adapter uses). One batched request
per call covers every held symbol, keeping the poll at one API hit per
reconciler tick — far inside Alpaca's data rate limits.

Deliberately NO retry loop here (unlike the trading adapter's
`_retry`): the news poll is best-effort event detection driven every
tick by the EventTriggerEngine, which already logs and swallows a
failed poll — the next tick (minutes away) is the retry. Blocking a
reconcile tick for up to five backoff attempts to fetch headlines would
invert the priority order.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, SecretStr

from observability.log_port import get_logger

log = get_logger(__name__)

# One page of the newest articles across all held symbols. Held books
# run ~10 names; 50 newest articles per 5-minute tick is comfortably
# lossless for event detection.
_DEFAULT_PAGE_LIMIT = 50


class NewsArticle(BaseModel):
    """Normalized article shape the EventTriggerEngine consumes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    article_id: str
    headline: str
    summary: str
    symbols: tuple[str, ...]
    created_at: dt.datetime


class AlpacaNewsClient:
    """`latest_news(symbols)` -> newest-first `NewsArticle` list."""

    def __init__(
        self,
        *,
        api_key_id: SecretStr,
        api_secret: SecretStr,
        page_limit: int = _DEFAULT_PAGE_LIMIT,
    ) -> None:
        # Late import so unit tests of consumers never need the SDK.
        from alpaca.data.historical.news import NewsClient

        self._client = NewsClient(
            api_key=api_key_id.get_secret_value(),
            secret_key=api_secret.get_secret_value(),
        )
        self._page_limit = page_limit

    def latest_news(self, symbols: Sequence[str]) -> list[NewsArticle]:
        """Newest-first articles mentioning any of `symbols`. One API
        request; no pagination — event detection only needs the newest
        page. Raises on transport errors (the caller logs + skips)."""
        if not symbols:
            return []
        from alpaca.data.requests import NewsRequest

        news_set = self._client.get_news(
            NewsRequest(
                symbols=",".join(symbols),
                limit=self._page_limit,
                # Contentless articles still carry headline + symbols —
                # exactly the trigger signal — so don't exclude them.
                exclude_contentless=False,
            )
        )
        raw_articles = getattr(news_set, "data", {}).get("news", [])
        return [_to_article(raw) for raw in raw_articles]


def _to_article(raw: object) -> NewsArticle:
    """Normalize an alpaca-py `News` model (or anything duck-shaped like
    one) into the engine's `NewsArticle`."""
    return NewsArticle(
        article_id=str(getattr(raw, "id", "")),
        headline=str(getattr(raw, "headline", "") or ""),
        summary=str(getattr(raw, "summary", "") or ""),
        symbols=tuple(getattr(raw, "symbols", ()) or ()),
        created_at=getattr(raw, "created_at", dt.datetime.now(dt.UTC)),
    )


__all__ = ["AlpacaNewsClient", "NewsArticle"]
