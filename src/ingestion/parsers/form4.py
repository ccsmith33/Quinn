"""Form 4 XML parser (S3.3 AC-3).

Extracts issuer, reporting owner, transactions, and footnote text from the
SEC's `ownershipDocument` schema. The structured output is what the LLM
analyzer (S5.3) sees in place of the raw XML — see ADR-002 §3.

We intentionally model only what the v1 prompts read; fields like
derivative tables, post-transaction beneficial ownership decomposition,
and signature blocks are skipped. When prompts evolve, this parser
extends rather than the on-disk format mutating silently.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from lxml import etree
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

# Defense-in-depth (carry-forward S3.2 review L-1): refuse external entity
# resolution and outbound DTD fetches. SEC Form 4 XML is trusted (egress
# is pinned to *.sec.gov per architecture §9.5), but tightened parser
# defaults are free hardening against any future supply-chain anomaly.
_SAFE_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


def _text(parent: etree._Element | None, path: str) -> str | None:
    """Return text under `path`, transparently unwrapping `<value>...</value>`."""
    if parent is None:
        return None
    el = parent.find(path)
    if el is None:
        return None
    inner = el.find("value")
    if inner is not None and inner.text:
        return inner.text.strip()
    if el.text:
        return el.text.strip()
    return None


def _int_text(parent: etree._Element | None, path: str) -> int | None:
    raw = _text(parent, path)
    if raw is None:
        return None
    return int(raw)


class Form4Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_title: str
    transaction_date: dt.date
    transaction_code: str
    shares: Decimal
    price_per_share: Decimal
    acquired_disposed: str  # "A" (acquired) | "D" (disposed)

    @field_validator("shares", "price_per_share", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @field_serializer("shares", "price_per_share")
    def _ser_decimal(self, v: Decimal) -> str:
        # Stable string representation; preserves trailing zeros that the
        # filer wrote (e.g., "5.25" vs "5.250000").
        return format(v, "f")

    @field_serializer("transaction_date")
    def _ser_date(self, v: dt.date) -> str:
        return v.isoformat()


class Form4Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issuer_cik: int
    issuer_name: str
    issuer_ticker: str
    reporting_owner_cik: int
    reporting_owner_name: str
    reporting_owner_relationship: str | None
    period_of_report: dt.date
    transactions: list[Form4Transaction]
    footnotes: str  # concatenated footnote text, in declaration order

    @field_serializer("period_of_report")
    def _ser_period(self, v: dt.date) -> str:
        return v.isoformat()

    def to_normalized_json(self) -> str:
        return self.model_dump_json(indent=None)

    @classmethod
    def from_normalized_json(cls, blob: str) -> Form4Document:
        return cls.model_validate_json(blob)


class Form4ParseError(Exception):
    """Raised when the XML is structurally not a Form 4 ownershipDocument
    or required fields are absent."""


def parse_form4_xml(xml_bytes: bytes) -> Form4Document:
    try:
        root = etree.fromstring(xml_bytes, parser=_SAFE_PARSER)
    except etree.XMLSyntaxError as e:
        raise Form4ParseError(f"invalid Form 4 XML: {e}") from e

    if root.tag != "ownershipDocument":
        raise Form4ParseError(f"expected ownershipDocument, got <{root.tag}>")

    issuer = root.find("issuer")
    issuer_cik = _int_text(issuer, "issuerCik")
    issuer_name = _text(issuer, "issuerName")
    issuer_ticker = _text(issuer, "issuerTradingSymbol")
    if issuer_cik is None or issuer_name is None or issuer_ticker is None:
        raise Form4ParseError("missing issuer fields")

    owner = root.find("reportingOwner")
    owner_id = owner.find("reportingOwnerId") if owner is not None else None
    owner_cik = _int_text(owner_id, "rptOwnerCik")
    owner_name = _text(owner_id, "rptOwnerName")
    if owner_cik is None or owner_name is None:
        raise Form4ParseError("missing reporting owner fields")

    rel_el = owner.find("reportingOwnerRelationship") if owner is not None else None
    relationship = _summarize_relationship(rel_el)

    period_text = _text(root, "periodOfReport")
    if period_text is None:
        raise Form4ParseError("missing periodOfReport")
    period = dt.date.fromisoformat(period_text)

    transactions: list[Form4Transaction] = []
    table = root.find("nonDerivativeTable")
    if table is not None:
        for tx_el in table.findall("nonDerivativeTransaction"):
            tx = _parse_transaction(tx_el)
            if tx is not None:
                transactions.append(tx)

    footnotes = _collect_footnotes(root.find("footnotes"))

    return Form4Document(
        issuer_cik=issuer_cik,
        issuer_name=issuer_name,
        issuer_ticker=issuer_ticker,
        reporting_owner_cik=owner_cik,
        reporting_owner_name=owner_name,
        reporting_owner_relationship=relationship,
        period_of_report=period,
        transactions=transactions,
        footnotes=footnotes,
    )


def _parse_transaction(tx_el: etree._Element) -> Form4Transaction | None:
    security_title = _text(tx_el, "securityTitle")
    tx_date_text = _text(tx_el, "transactionDate")
    coding = tx_el.find("transactionCoding")
    code = _text(coding, "transactionCode") if coding is not None else None
    amounts = tx_el.find("transactionAmounts")
    shares_text = _text(amounts, "transactionShares") if amounts is not None else None
    price_text = (
        _text(amounts, "transactionPricePerShare") if amounts is not None else None
    )
    ad_text = (
        _text(amounts, "transactionAcquiredDisposedCode")
        if amounts is not None
        else None
    )
    if (
        security_title is None
        or tx_date_text is None
        or code is None
        or shares_text is None
        or price_text is None
        or ad_text is None
    ):
        return None
    return Form4Transaction(
        security_title=security_title,
        transaction_date=dt.date.fromisoformat(tx_date_text),
        transaction_code=code,
        shares=Decimal(shares_text),
        price_per_share=Decimal(price_text),
        acquired_disposed=ad_text,
    )


def _summarize_relationship(rel_el: etree._Element | None) -> str | None:
    if rel_el is None:
        return None
    parts: list[str] = []
    for tag in ("isDirector", "isOfficer", "isTenPercentOwner", "isOther"):
        sub = rel_el.find(tag)
        if sub is not None and (sub.text or "").strip() in {"1", "true"}:
            parts.append(tag)
    title = _text(rel_el, "officerTitle")
    if title:
        parts.append(title)
    return ", ".join(parts) if parts else None


def _collect_footnotes(fn_el: etree._Element | None) -> str:
    if fn_el is None:
        return ""
    pieces: list[str] = []
    for fn in fn_el.findall("footnote"):
        if fn.text:
            pieces.append(fn.text.strip())
    return "\n".join(pieces)
