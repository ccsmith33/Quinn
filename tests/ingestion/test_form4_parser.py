"""S3.3 — Form 4 XML parser.

ADR-002 §3 test scenario: "Ingestion of a Form 4 with footnoted transaction
codes; verify the structured fields parse correctly and the footnote text is
preserved for the LLM."
"""

from __future__ import annotations

from decimal import Decimal

from ingestion.parsers.form4 import Form4Document, Form4Transaction, parse_form4_xml

# Real-shape SEC Form 4 XML (anonymized, single non-derivative transaction
# with one footnote reference).
_FORM4_FIXTURE = b"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0306</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-04-29</periodOfReport>
  <issuer>
    <issuerCik>0001234567</issuerCik>
    <issuerName>ACME MICROCAP CORP</issuerName>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0009999000</rptOwnerCik>
      <rptOwnerName>SMITH JOHN</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-04-28</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare>
          <value>5.25</value><footnoteId id="F1"/>
        </transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote id="F1">Average price; range $5.20-$5.30 per share.</footnote>
    <footnote id="F2">Held indirectly by family trust.</footnote>
  </footnotes>
</ownershipDocument>
"""


def test_form4_xml_parsed_with_footnotes() -> None:
    doc = parse_form4_xml(_FORM4_FIXTURE)
    assert isinstance(doc, Form4Document)
    assert doc.issuer_cik == 1234567
    assert doc.issuer_name == "ACME MICROCAP CORP"
    assert doc.issuer_ticker == "ACME"
    assert doc.reporting_owner_cik == 9999000
    assert doc.reporting_owner_name == "SMITH JOHN"
    assert "Chief Executive Officer" in (doc.reporting_owner_relationship or "")

    assert len(doc.transactions) == 1
    tx: Form4Transaction = doc.transactions[0]
    assert tx.transaction_code == "P"
    assert tx.shares == Decimal("10000")
    assert tx.price_per_share == Decimal("5.25")
    assert tx.transaction_date.isoformat() == "2026-04-28"
    assert tx.acquired_disposed == "A"
    assert tx.security_title == "Common Stock"

    # Both footnotes preserved (concatenated for LLM).
    assert "Average price" in doc.footnotes
    assert "$5.20" in doc.footnotes
    assert "Held indirectly" in doc.footnotes


def test_form4_to_normalized_json_roundtrip() -> None:
    """The on-disk representation: a stable, sorted-keys JSON string the LLM
    will see in place of the raw XML."""
    doc = parse_form4_xml(_FORM4_FIXTURE)
    blob = doc.to_normalized_json()
    assert isinstance(blob, str)
    # Deterministic: re-parsing yields equal object.
    doc2 = Form4Document.from_normalized_json(blob)
    assert doc2 == doc


def test_form4_handles_no_footnotes() -> None:
    payload = b"""<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-04-29</periodOfReport>
  <issuer>
    <issuerCik>0000000001</issuerCik>
    <issuerName>ZED CO</issuerName>
    <issuerTradingSymbol>ZED</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000000002</rptOwnerCik>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common</value></securityTitle>
      <transactionDate><value>2026-04-28</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>10</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""
    doc = parse_form4_xml(payload)
    assert doc.footnotes == ""
    assert doc.transactions[0].transaction_code == "S"
