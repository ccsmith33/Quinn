"""Routine 8-K rules — drop filings whose item-code structure makes
them ~never tradeable BEFORE the LLM ever sees them. Pure logic; codes
+ raw filing body text; deterministic.

See task #1 (cost-cuts day 4): Sonnet/Haiku reasoning on these patterns
verbatim "routine governance, no material economic consequence" /
"purely procedural, no immediate operational impact" / "standard,
recurring distribution announcement".
"""

from __future__ import annotations

from prefilter.routine_8k import RoutineEightKFilter


# ---------------------------------------------------------------------------
# Definite drops — item codes are exactly {5.07} (annual meeting voting)
# ---------------------------------------------------------------------------


def test_5_07_alone_dropped() -> None:
    """Pure 5.07 (annual meeting voting): never produces trades."""
    decision = RoutineEightKFilter().evaluate(["5.07"], raw_text="any text")
    assert decision is not None
    assert decision.rule_fired == "routine_5_07_voting"


def test_5_07_with_9_01_dropped() -> None:
    """9.01 is exhibit-list metadata; doesn't change the routine nature."""
    decision = RoutineEightKFilter().evaluate(
        ["5.07", "9.01"], raw_text="any text"
    )
    assert decision is not None
    assert decision.rule_fired == "routine_5_07_voting"


def test_5_07_plus_substantive_item_not_dropped() -> None:
    """1.01 (material agreement) trumps 5.07 — keep going to the LLM."""
    decision = RoutineEightKFilter().evaluate(
        ["1.01", "5.07"], raw_text="any text"
    )
    assert decision is None


def test_2_02_with_5_07_not_dropped() -> None:
    """Earnings (2.02) mixed with 5.07: 2.02 is the catalyst, never drop."""
    decision = RoutineEightKFilter().evaluate(
        ["2.02", "5.07", "9.01"], raw_text="any text"
    )
    assert decision is None


# ---------------------------------------------------------------------------
# Definite drops — {5.02} retirement (no successor named, no economic terms)
# ---------------------------------------------------------------------------


def test_5_02_retirement_dropped_when_no_appointment_terms() -> None:
    """{5.02} alone with body containing only retirement/departing terms
    and NO 'appointed' or 'elected' (no incoming officer)."""
    body = "Jane Doe is retiring as Chief Financial Officer effective May 31."
    decision = RoutineEightKFilter().evaluate(["5.02"], raw_text=body)
    assert decision is not None
    assert decision.rule_fired == "routine_5_02_retirement"


def test_5_02_with_9_01_retirement_dropped() -> None:
    body = "Departing executive — John Smith resigns as Chief Operating Officer."
    decision = RoutineEightKFilter().evaluate(["5.02", "9.01"], raw_text=body)
    assert decision is not None
    assert decision.rule_fired == "routine_5_02_retirement"


def test_5_02_with_appointment_keyword_not_dropped() -> None:
    """If the body contains 'appointed' or 'elected', a new officer is
    coming in — economic terms may be material; do NOT drop."""
    body_appointed = "Jane Doe was appointed as the new Chief Financial Officer."
    assert (
        RoutineEightKFilter().evaluate(["5.02"], raw_text=body_appointed) is None
    )
    body_elected = "Mr. Smith was elected to the Board of Directors."
    assert (
        RoutineEightKFilter().evaluate(["5.02"], raw_text=body_elected) is None
    )


def test_5_02_plus_substantive_item_not_dropped() -> None:
    body = "Jane Doe is retiring as CFO."
    decision = RoutineEightKFilter().evaluate(
        ["1.01", "5.02"], raw_text=body
    )
    assert decision is None


# ---------------------------------------------------------------------------
# Probable drops — {8.01} routine dividend / distribution
# ---------------------------------------------------------------------------


def test_8_01_routine_dividend_dropped() -> None:
    body = (
        "The Board of Directors declared a quarterly cash dividend of "
        "$0.42 per share, payable June 15, 2026."
    )
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    assert decision is not None
    assert decision.rule_fired == "routine_8_01_dividend"


def test_8_01_routine_distribution_dropped() -> None:
    """BDC monthly distribution — same pattern."""
    body = "The Company declared a regular monthly distribution of $0.075 per share."
    decision = RoutineEightKFilter().evaluate(
        ["8.01", "9.01"], raw_text=body
    )
    assert decision is not None
    assert decision.rule_fired == "routine_8_01_dividend"


def test_8_01_dividend_with_guidance_not_dropped() -> None:
    """Guidance content is substantive — don't drop even if the headline
    is a dividend."""
    body = (
        "Declared a $0.42 dividend. The Company is also raising full-year "
        "guidance for revenue to a range of $4.2 to $4.4 billion."
    )
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    assert decision is None


def test_8_01_dividend_with_agreement_not_dropped() -> None:
    body = (
        "Declared a quarterly dividend of $0.50 per share. Separately, the "
        "Company entered into a credit agreement with its lenders."
    )
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    assert decision is None


def test_8_01_no_dividend_keyword_not_dropped() -> None:
    """Plain 8.01 without dividend pattern shouldn't be dropped — could be
    unusual news. The rule is conservative."""
    body = "The Company today announced a strategic partnership with Acme Corp."
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    assert decision is None


def test_8_01_dividend_no_numeric_amount_not_dropped() -> None:
    """Dividend keyword without a numeric amount near it could be a
    discussion / policy change — don't drop."""
    body = "The Board reviewed the Company's dividend policy at its quarterly meeting."
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    assert decision is None


# ---------------------------------------------------------------------------
# Probable drops — {7.01} IR/conference posting (no Reg FD substance)
# ---------------------------------------------------------------------------


def test_7_01_investor_presentation_dropped() -> None:
    body = (
        "The Company is furnishing its investor presentation that will be used "
        "at upcoming meetings with investors and analysts."
    )
    decision = RoutineEightKFilter().evaluate(["7.01"], raw_text=body)
    assert decision is not None
    assert decision.rule_fired == "routine_7_01_ir_conference"


def test_7_01_conference_dropped() -> None:
    body = (
        "On May 12, 2026, the Company will participate in a fireside chat "
        "at the Acme Capital Markets Conference."
    )
    decision = RoutineEightKFilter().evaluate(
        ["7.01", "9.01"], raw_text=body
    )
    assert decision is not None
    assert decision.rule_fired == "routine_7_01_ir_conference"


def test_7_01_with_reg_fd_substance_not_dropped() -> None:
    """If the body explicitly invokes 'Regulation FD', there is substantive
    selective-disclosure content — do not drop."""
    body = (
        "Pursuant to Regulation FD, the Company is providing the following "
        "preliminary financial information for Q2."
    )
    decision = RoutineEightKFilter().evaluate(["7.01"], raw_text=body)
    assert decision is None


def test_7_01_neither_pattern_not_dropped() -> None:
    """Plain 7.01 without IR-presentation / conference keywords — don't drop."""
    body = "The Company today made the following statement to clarify recent media reports."
    decision = RoutineEightKFilter().evaluate(["7.01"], raw_text=body)
    assert decision is None


# ---------------------------------------------------------------------------
# Substantive items always survive
# ---------------------------------------------------------------------------


def test_1_01_alone_not_dropped() -> None:
    """1.01 (material definitive agreement) is never routine."""
    assert RoutineEightKFilter().evaluate(["1.01"], raw_text="any text") is None


def test_2_01_alone_not_dropped() -> None:
    """2.01 (completion of acquisition) is never routine."""
    assert RoutineEightKFilter().evaluate(["2.01"], raw_text="any text") is None


def test_2_02_alone_not_dropped() -> None:
    """2.02 (earnings) is never routine."""
    assert RoutineEightKFilter().evaluate(["2.02"], raw_text="any text") is None


def test_empty_codes_not_dropped() -> None:
    """The orchestrator handles empty codes via the existing item_code_empty
    branch; this filter is a no-op on empty input."""
    assert RoutineEightKFilter().evaluate([], raw_text="any text") is None


def test_keyword_match_is_case_insensitive() -> None:
    """Filings vary in casing — 'DIVIDEND' should match 'dividend'."""
    body = "DECLARED A QUARTERLY CASH DIVIDEND OF $0.50 PER SHARE."
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    assert decision is not None
    assert decision.rule_fired == "routine_8_01_dividend"


# ---------------------------------------------------------------------------
# Body-scan bound — prefilter must stay O(constant), not O(filing size).
# Post-task-#3, raw_text can run to ~1 MB; the routine filter scans only
# the leading 16 KB. Keywords past that cap must NOT match.
# ---------------------------------------------------------------------------


def test_dividend_keyword_past_scan_cap_not_matched() -> None:
    """A dividend declaration buried at byte 1M should NOT trigger the
    routine drop. Catches the case where an analyzed exhibit appended to
    raw_text contains an unrelated dividend reference far past the cover
    page."""
    from prefilter.routine_8k import _BODY_SCAN_BYTES

    filler = "x" * (_BODY_SCAN_BYTES * 4)
    body = (
        "The Company is making the following routine clarification. "
        + filler
        + " Declared a quarterly cash dividend of $0.42 per share."
    )
    decision = RoutineEightKFilter().evaluate(["8.01"], raw_text=body)
    # Cover page has no dividend pattern; the dividend phrase is past
    # the cap → no match → no drop. The LLM still gets to see it.
    assert decision is None


def test_appointment_keyword_past_scan_cap_not_seen_so_drops() -> None:
    """Conversely: 5.02 retirement looks for 'appointed'/'elected' as a
    DISQUALIFIER. If those terms only appear past the scan cap, the
    filter cannot see them — and the filing drops as routine retirement.
    This is intentional: the cover page is where new-officer-economic-
    terms live; an 'appointed' buried at byte 1M is overwhelmingly an
    exhibit boilerplate artefact (proxy materials), not the catalyst.
    """
    from prefilter.routine_8k import _BODY_SCAN_BYTES

    filler = "x" * (_BODY_SCAN_BYTES * 4)
    body = (
        "Jane Doe is retiring as Chief Financial Officer effective May 31. "
        + filler
        + " John Smith was appointed to a board committee."
    )
    decision = RoutineEightKFilter().evaluate(["5.02"], raw_text=body)
    assert decision is not None
    assert decision.rule_fired == "routine_5_02_retirement"


def test_routine_filter_handles_one_mb_body_without_crashing() -> None:
    """Smoke check: invoking the filter with a 1 MB body completes
    promptly. Not a timing assertion (would be flaky in CI) — just
    confirms the function returns and doesn't accidentally re-scan the
    full body. The slice + single .lower() makes this trivial."""
    from prefilter.routine_8k import _BODY_SCAN_BYTES

    big_body = "x" * (_BODY_SCAN_BYTES * 64)  # ~1 MB
    assert RoutineEightKFilter().evaluate(["8.01"], raw_text=big_body) is None
