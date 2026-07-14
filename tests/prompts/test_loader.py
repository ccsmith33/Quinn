"""S5.1 — Prompt loader tests.

Architecture references: §3.1 (prompt versioning), §3.2..§3.4 (schemas),
ADR-005 (content-hash version ids, three-block cache structure),
FR-29, NFR-12.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import pytest

from journal.models import FilingRow, ProposalRow
from prompts.loader import (
    _FILING_BODY_BYTE_CAP,
    _FILING_BODY_TRUNCATION_MARKER,
    AnalyzerContext,
    ApiRequest,
    PromptBuilder,
)
from prompts.lock import verify_lock, write_lock

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_prompt_dir(root: Path) -> Path:
    """Build a minimal prompt dir with the structure ADR-005 requires."""
    (root / "fragments").mkdir(parents=True)
    (root / "schemas").mkdir()

    # Top-level prompts. The v2s exist because the builder's ACTIVE_*
    # names (D-079) compose them; v1s stay for the version-roll tests.
    (root / "sonnet_filing_analysis_v1.txt").write_text(
        "# Sonnet filing analysis prompt v1\n"
        "Include: <<role>> <<rules_invariant>> <<output_schema>>\n"
        "Schema: <<schemas/trade_proposal.json>>\n"
    )
    (root / "sonnet_filing_analysis_v2.txt").write_text(
        "# Sonnet filing analysis prompt v2\n"
        "Include: <<role>> <<rules_invariant>> <<output_schema>>\n"
        "Schema: <<schemas/trade_proposal_v2.json>>\n"
    )
    (root / "opus_proposal_review_v1.txt").write_text(
        "# Opus proposal review prompt v1\n"
        "Include: <<role>> <<rules_invariant>>\n"
        "Schema: <<schemas/opus_proposal_review.json>>\n"
    )
    (root / "opus_proposal_review_v2.txt").write_text(
        "# Opus proposal review prompt v2\n"
        "Include: <<role>> <<rules_invariant>>\n"
        "Schema: <<schemas/opus_proposal_review.json>>\n"
    )

    # Fragments referenced by both.
    (root / "fragments" / "role.txt").write_text(
        "You are an autonomous swing-trading analyst.\n"
    )
    (root / "fragments" / "rules_invariant.txt").write_text(
        "Rules: long-only; conviction 1..10; size_pct ≤ 0.20.\n"
    )
    (root / "fragments" / "output_schema.txt").write_text(
        "Emit a JSON object conforming to the schema.\n"
    )

    # Schemas (minimal placeholders; real content comes verbatim from arch §3).
    (root / "schemas" / "trade_proposal.json").write_text(
        json.dumps({"title": "TradeProposal", "type": "object"})
    )
    (root / "schemas" / "no_trade_record.json").write_text(
        json.dumps({"title": "NoTradeRecord", "type": "object"})
    )
    (root / "schemas" / "opus_proposal_review.json").write_text(
        json.dumps({"title": "OpusProposalReview", "type": "object"})
    )
    (root / "schemas" / "trade_proposal_v2.json").write_text(
        json.dumps({"title": "TradeProposal", "type": "object"})
    )
    return root


def _filing(**overrides: object) -> FilingRow:
    base = dict(
        id=42,
        accession_number="0001234567-26-000001",
        cik=1234567,
        form_type="8-K",
        filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
        fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
        raw_text_path="/var/lib/quinn/raw/0001234567-26-000001.txt",
        content_hash="abc123",
        item_codes='["1.01"]',
        issuer_ticker="ACME",
        ingest_state="ok",
        ingest_error=None,
    )
    base.update(overrides)
    return FilingRow(**base)


def _ctx() -> AnalyzerContext:
    return AnalyzerContext(
        universe_summary="ACME (NASDAQ, mcap=500M, prev_close=12.34)",
        kill_switch_state="active",
        open_positions_count=0,
        decision_id="dec_xyz_001",
    )


def _proposal(**overrides: object) -> ProposalRow:
    base = dict(
        id=99,
        filing_id=42,
        decision_id="dec_xyz_001",
        model_id="claude-sonnet-4-6",
        prompt_version="sonnet_filing_analysis_v1@aaaaaaaaaaaa",
        raw_response='{"conviction": 8}',
        kind="trade_proposal",
        symbol="ACME",
        direction="long",
        size_pct_requested=0.05,
        conviction=8,
        thesis="strong material 8-K item 2.02",
        input_tokens=1000,
        output_tokens=200,
        latency_ms=1234,
        cost_usd=0.012,
    )
    base.update(overrides)
    return ProposalRow(**base)


# ---------------------------------------------------------------------------
# AC-3: prompt_version is deterministic content-hash
# ---------------------------------------------------------------------------

def test_prompt_version_is_deterministic_hash(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder1 = PromptBuilder(pdir)
    builder2 = PromptBuilder(pdir)

    v1 = builder1.prompt_version("sonnet_filing_analysis_v1")
    v2 = builder2.prompt_version("sonnet_filing_analysis_v1")
    assert v1 == v2  # same files → same id

    # format: name@12-hex-chars
    name, _, digest = v1.partition("@")
    assert name == "sonnet_filing_analysis_v1"
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# AC-7 / ADR-005: edit prompt → new version
# ---------------------------------------------------------------------------

def test_modify_file_changes_version(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    v_before = PromptBuilder(pdir).prompt_version("sonnet_filing_analysis_v1")

    (pdir / "sonnet_filing_analysis_v1.txt").write_text(
        "# Sonnet filing analysis prompt v1 — EDITED\n"
        "Include: <<role>> <<rules_invariant>> <<output_schema>>\n"
        "Schema: <<schemas/trade_proposal.json>>\n"
    )
    v_after = PromptBuilder(pdir).prompt_version("sonnet_filing_analysis_v1")
    assert v_before != v_after


# ---------------------------------------------------------------------------
# AC-7 / ADR-005: modify fragment → top-level version changes too
# ---------------------------------------------------------------------------

def test_modify_fragment_changes_top_level_version(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    sonnet_before = PromptBuilder(pdir).prompt_version("sonnet_filing_analysis_v1")
    opus_before = PromptBuilder(pdir).prompt_version("opus_proposal_review_v1")

    # Both prompts include role.txt — modifying it must roll into both.
    (pdir / "fragments" / "role.txt").write_text(
        "You are an autonomous swing-trading analyst — UPDATED.\n"
    )
    sonnet_after = PromptBuilder(pdir).prompt_version("sonnet_filing_analysis_v1")
    opus_after = PromptBuilder(pdir).prompt_version("opus_proposal_review_v1")

    assert sonnet_before != sonnet_after
    assert opus_before != opus_after


# ---------------------------------------------------------------------------
# AC-4: system block bytes stable across calls with different filings
# ---------------------------------------------------------------------------

def test_system_block_bytes_stable_across_filings(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    ctx = _ctx()

    req_a = builder.build_sonnet_filing_analysis(_filing(), "filing payload A", ctx)
    req_b = builder.build_sonnet_filing_analysis(
        _filing(accession_number="0001234567-26-000999", cik=2222222),
        "filing payload B — completely different",
        ctx,
    )

    # Block 1 (system) must be byte-identical.
    assert req_a.system == req_b.system
    # Block 3 (filing payload in messages) must differ.
    assert req_a.messages != req_b.messages


# ---------------------------------------------------------------------------
# AC-2: cache_control markers placed correctly per ADR-005
# ---------------------------------------------------------------------------

def test_cache_control_placement(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    req = builder.build_sonnet_filing_analysis(_filing(), "filing payload", _ctx())

    # Block 1 (system) and Block 2 (universe + decision context) live in `system`.
    # Block 1 is cached; block 2 is cached. Block 3 (filing payload) lives in
    # messages and is NOT cached.
    assert isinstance(req, ApiRequest)
    assert len(req.system) >= 2  # at least block 1 and block 2

    # The last entry of block 1 carries cache_control: ephemeral.
    # Block 1 ends and block 2 begins; the last item of each carries the marker.
    cache_marked = [b for b in req.system if b.cache_control is not None]
    assert len(cache_marked) == 2  # block 1 boundary + block 2 boundary
    for b in cache_marked:
        assert b.cache_control == {"type": "ephemeral"}

    # No cache_control on any message content.
    for msg in req.messages:
        for c in msg.content:
            assert c.cache_control is None


# ---------------------------------------------------------------------------
# AC-1: build_opus_proposal_review returns ApiRequest
# ---------------------------------------------------------------------------

def test_build_opus_proposal_review(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    req = builder.build_opus_proposal_review(_proposal(), "summary of source filing text")

    assert isinstance(req, ApiRequest)
    # Same three-block structure: at least 2 system blocks, 1 user message.
    assert len(req.system) >= 2
    assert len(req.messages) >= 1
    # cache_control on system block boundaries (per ADR-005 §"Prompt structure").
    cache_marked = [b for b in req.system if b.cache_control is not None]
    assert len(cache_marked) == 2
    # Filing payload in user message reflects proposal under review.
    user_text = "".join(c.text for m in req.messages for c in m.content)
    assert "summary of source filing text" in user_text


# ---------------------------------------------------------------------------
# AC-5: prompts.lock matches files on disk
# ---------------------------------------------------------------------------

def test_lock_file_matches_files_on_disk(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    lock_path = pdir / "prompts.lock"

    write_lock(pdir, lock_path)
    verify_lock(pdir, lock_path)  # should not raise

    # Tamper a file → verify_lock raises.
    (pdir / "fragments" / "role.txt").write_text("# tampered\n")
    with pytest.raises(Exception):  # noqa: B017 — generic LockMismatch
        verify_lock(pdir, lock_path)


def test_lock_file_format(tmp_path: Path) -> None:
    pdir = _make_prompt_dir(tmp_path / "prompts")
    lock_path = pdir / "prompts.lock"
    write_lock(pdir, lock_path)

    content = lock_path.read_text()
    # one line per *.txt or *.json file, sorted, format: "<relpath>  <sha256>"
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) >= 5  # 2 top + 3 fragments + 3 schemas = 8 total

    # Validate at least one known line.
    role_path = pdir / "fragments" / "role.txt"
    expected_hash = hashlib.sha256(role_path.read_bytes()).hexdigest()
    found = [line for line in lines if "fragments/role.txt" in line]
    assert len(found) == 1
    assert expected_hash in found[0]


# ---------------------------------------------------------------------------
# AC-6: project ships actual prompt files at expected paths
# ---------------------------------------------------------------------------

def test_real_prompt_files_load(tmp_path: Path) -> None:
    """The committed `src/prompts/` files load and produce a request shape.

    Uses the actual `src/prompts/` dir, not a fixture, to verify AC-6.
    """
    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    assert (pdir / "sonnet_filing_analysis_v1.txt").is_file()
    assert (pdir / "opus_proposal_review_v1.txt").is_file()
    assert (pdir / "fragments").is_dir()
    assert (pdir / "schemas" / "trade_proposal.json").is_file()
    assert (pdir / "schemas" / "no_trade_record.json").is_file()
    assert (pdir / "schemas" / "opus_proposal_review.json").is_file()

    builder = PromptBuilder(pdir)
    req = builder.build_sonnet_filing_analysis(_filing(), "raw filing text", _ctx())
    assert isinstance(req, ApiRequest)
    assert builder.prompt_version("sonnet_filing_analysis_v1").startswith(
        "sonnet_filing_analysis_v1@"
    )


def test_d079_v2_prompts_ship_and_v2_is_active_for_all_stages() -> None:
    """D-079 §3.2/§3.4: the v2 prompt files exist beside byte-identical
    v1s (ADR-005 — new versions, never in-place edits) and ALL stages
    compose v2. The sonnet flip landed at integration together with
    loop._compose_decision_id importing ACTIVE_SONNET_ANALYSIS_PROMPT,
    so the decision-id mirror moves with the constant by construction."""
    from prompts.loader import (
        ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT,
        ACTIVE_OPUS_THESIS_REVIEW_PROMPT,
        ACTIVE_SONNET_ANALYSIS_PROMPT,
    )

    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    for name in (
        "sonnet_filing_analysis_v2",
        "opus_proposal_review_v2",
        "opus_thesis_review_v2",
    ):
        assert (pdir / f"{name}.txt").is_file()
    assert (pdir / "schemas" / "thesis_review_v2.json").is_file()
    assert (pdir / "schemas" / "trade_proposal_v2.json").is_file()

    assert ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT == "opus_proposal_review_v2"
    assert ACTIVE_OPUS_THESIS_REVIEW_PROMPT == "opus_thesis_review_v2"
    assert ACTIVE_SONNET_ANALYSIS_PROMPT == "sonnet_filing_analysis_v2"

    builder = PromptBuilder(pdir)
    # The composed thesis-review v2 carries the new decision and the v2
    # schema (adjust_take_profit / new_tp_price).
    composed = builder._composed_system_bytes(  # noqa: SLF001
        "opus_thesis_review_v2"
    ).decode("utf-8")
    assert "adjust_take_profit" in composed
    assert "new_tp_price" in composed
    # Symmetric exit guidance reached the proposal reviewer.
    review_composed = builder._composed_system_bytes(  # noqa: SLF001
        "opus_proposal_review_v2"
    ).decode("utf-8")
    assert "widen" in review_composed
    # The sonnet v2 states the R:R floor and the trail field.
    sonnet_composed = builder._composed_system_bytes(  # noqa: SLF001
        "sonnet_filing_analysis_v2"
    ).decode("utf-8")
    assert "1.5" in sonnet_composed
    assert "trail_distance_pct" in sonnet_composed


def test_committed_lock_matches_committed_files() -> None:
    """The on-disk `src/prompts/prompts.lock` matches the files in the same dir."""
    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    lock = pdir / "prompts.lock"
    assert lock.is_file(), "prompts.lock missing — run write_lock to regenerate"
    verify_lock(pdir, lock)


# ---------------------------------------------------------------------------
# Body-cap: prevent 200K-token context overruns on large 10-Qs.
# ---------------------------------------------------------------------------

def test_filing_body_under_cap_passes_through_unchanged(tmp_path: Path) -> None:
    """raw_text smaller than the cap is embedded verbatim, no truncation marker."""
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    body = "small filing body" * 100  # well under cap
    req = builder.build_sonnet_filing_analysis(_filing(), body, _ctx())

    user_text = "".join(c.text for m in req.messages for c in m.content)
    assert body in user_text
    assert _FILING_BODY_TRUNCATION_MARKER not in user_text


def test_filing_body_at_cap_is_truncated_with_marker(tmp_path: Path) -> None:
    """raw_text exceeding the cap is truncated and gets the explicit marker."""
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    body = "X" * (_FILING_BODY_BYTE_CAP + 50_000)
    req = builder.build_sonnet_filing_analysis(_filing(), body, _ctx())

    user_text = "".join(c.text for m in req.messages for c in m.content)
    assert _FILING_BODY_TRUNCATION_MARKER in user_text
    # The body that reaches the model must be ≤ cap + marker length (the
    # other surrounding metadata in the payload is small and doesn't matter
    # for the 200K-token guarantee — what matters is the body slice).
    body_in_payload = user_text.split("---\n", 1)[1]
    truncated_body = body_in_payload.rsplit(_FILING_BODY_TRUNCATION_MARKER, 1)[0]
    assert len(truncated_body.encode("utf-8")) <= _FILING_BODY_BYTE_CAP


def test_filing_body_truncation_logs_event(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When the cap binds, an info-level event is logged for operator visibility."""
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    body = "X" * (_FILING_BODY_BYTE_CAP + 1000)
    with caplog.at_level(logging.INFO, logger="prompts.loader"):
        builder.build_sonnet_filing_analysis(_filing(), body, _ctx())

    matching = [
        r for r in caplog.records
        if getattr(r, "event", None) == "filing_body_truncated_at_cap"
    ]
    assert matching, (
        "expected filing_body_truncated_at_cap event; "
        f"got: {[getattr(r, 'event', None) for r in caplog.records]}"
    )
    rec = matching[0]
    assert rec.levelno == logging.INFO
    assert rec.cap == _FILING_BODY_BYTE_CAP
    assert rec.original_size == len(body.encode("utf-8"))


def test_filing_body_under_cap_emits_no_truncation_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No log event when the body is below the cap."""
    pdir = _make_prompt_dir(tmp_path / "prompts")
    builder = PromptBuilder(pdir)
    body = "small body"
    with caplog.at_level(logging.INFO, logger="prompts.loader"):
        builder.build_sonnet_filing_analysis(_filing(), body, _ctx())

    matching = [
        r for r in caplog.records
        if getattr(r, "event", None) == "filing_body_truncated_at_cap"
    ]
    assert not matching


# ---------------------------------------------------------------------------
# Event-reviews doctrine: zone-aware thesis-review context
# ---------------------------------------------------------------------------


def _thesis_review_kwargs(**overrides: object) -> dict:
    base: dict = dict(
        proposal=_proposal(),
        execution_id=7,
        days_held=12,
        current_price=118.0,
        pct_change_since_entry=0.18,
        realized_fill_price=100.0,
        realized_dollar_size=10_000.0,
        time_horizon_days=14,
        stop_loss_price=95.0,
        take_profit_price=150.0,
        filings_since_entry_summary="(no filings since entry)",
    )
    base.update(overrides)
    return base


def test_build_opus_thesis_review_renders_zone_block_and_base_rates() -> None:
    """Every review (calendar AND event) carries the structured position
    state block: zone, armed status, HWM gain, current gain, days held,
    trigger reason (+ detail), and the hardcoded base-rates block."""
    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    builder = PromptBuilder(pdir)
    req = builder.build_opus_thesis_review(
        **_thesis_review_kwargs(
            position_zone=3,
            trail_armed=True,
            hwm_gain_pct=31.5,
            trigger_reason="event:filing:0001234567-26-000001",
            trigger_detail="filing form=8-K acc=0001234567-26-000001",
        )
    )
    user_text = "".join(c.text for m in req.messages for c in m.content)
    assert "# Position state and jurisdiction" in user_text
    assert "position_zone: 3" in user_text
    assert "trail_armed: true" in user_text
    assert "hwm_gain_pct: +31.50%" in user_text
    assert "current_gain_pct: +18.00%" in user_text
    assert "days_held: 12" in user_text
    assert "review_trigger: event:filing:0001234567-26-000001" in user_text
    assert "trigger_detail: filing form=8-K" in user_text
    assert "# Base rates (measured on this strategy's own book)" in user_text
    assert "P(reach +50% | reached +30%) ~= 54%" in user_text
    assert "P(double | reached +50%) ~= 38%" in user_text
    assert "Harvest-early policies scored WORST" in user_text


def test_build_opus_thesis_review_zone1_calendar_shape() -> None:
    """A calendar review renders the block too (zone 1, trigger 'entry',
    no detail line when none applies)."""
    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    builder = PromptBuilder(pdir)
    req = builder.build_opus_thesis_review(
        **_thesis_review_kwargs(
            position_zone=1,
            trail_armed=False,
            hwm_gain_pct=4.0,
            trigger_reason="entry",
            trigger_detail=None,
        )
    )
    user_text = "".join(c.text for m in req.messages for c in m.content)
    assert "position_zone: 1" in user_text
    assert "trail_armed: false" in user_text
    assert "review_trigger: entry" in user_text
    assert "trigger_detail:" not in user_text


def test_build_opus_thesis_review_legacy_shape_unchanged_without_zone() -> None:
    """Callers that pass no zone kwargs (legacy shape) get the
    pre-doctrine payload — no jurisdiction or base-rates block."""
    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    builder = PromptBuilder(pdir)
    req = builder.build_opus_thesis_review(**_thesis_review_kwargs())
    user_text = "".join(c.text for m in req.messages for c in m.content)
    assert "# Position state and jurisdiction" not in user_text
    assert "# Base rates" not in user_text


def test_thesis_review_v2_prompt_carries_jurisdiction_doctrine() -> None:
    """The active thesis-review prompt states the zone doctrine: Zone 1
    judgment-primary, Zone 3 algorithm-primary with information events
    required (and price-shape reasoning banned), one-directional powers."""
    pdir = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
    text = (pdir / "opus_thesis_review_v2.txt").read_text(encoding="utf-8")
    assert "# Jurisdiction zones" in text
    assert "NEW INFORMATION" in text
    assert "looks toppy" in text  # the ban is stated explicitly
    assert "ONE-DIRECTIONAL" in text
    assert "Information events pierce this exemption" in text
