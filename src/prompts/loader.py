"""Prompt loader — composes file fragments into Anthropic request shape.

Architecture references: §3.1 (versioning), §3.2..§3.4 (schemas),
ADR-005 (per-file content-hash; three-block cache structure), FR-29, NFR-12.

Per D-031: this loader is pure (no DB writes). Composed prompt-version ids
emitted here coexist with S1.5's file-level ids in the journal `prompts`
table. Downstream wiring (S5.2 / agent main) registers composed ids before
journaling proposals.

Per D-032: `prompts.lock` lives at `src/prompts/prompts.lock`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from journal.models import FilingRow, ProposalRow
from observability.log_port import get_logger

_HASH_PREFIX_LEN = 12

# 400KB ≈ 100K tokens — leaves headroom for system blocks + prompt scaffolding
# under Anthropic's 200K context limit. Applied at prompt-build time so the
# persisted raw filing on disk is preserved for audit/replay; only what the
# model sees is truncated. Same pattern as the exhibit cap (ADR-008).
_FILING_BODY_BYTE_CAP = 400_000
_FILING_BODY_TRUNCATION_MARKER = "\n[...TRUNCATED AT BODY CAP...]"

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed request shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    """A single content block — text plus optional cache_control marker.

    Mirrors the Anthropic content-block shape consumed by the SDK in S5.2.
    """

    text: str
    cache_control: dict[str, str] | None = None


@dataclass(frozen=True)
class Message:
    role: str
    content: list[Block]


@dataclass(frozen=True)
class ApiRequest:
    """Composed request: system list + messages list, per ADR-005."""

    system: list[Block]
    messages: list[Message]
    prompt_version: str
    fragment_versions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalyzerContext:
    """Block-2 inputs: universe + decision context (cached, daily-stable).

    `kill_switch_state` is the v1 vocabulary literal — tightened from `str`
    per S5.6 carry-forward (S5.1 reviewer A-1). The analyzer prompt is
    written against these two values; any other value would be a wiring
    bug.
    """

    universe_summary: str
    kill_switch_state: Literal["halted", "ok"]
    open_positions_count: int
    decision_id: str


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

# Top-level prompt → list of (kind, ref) for ordered composition.
# kind="fragment" → fragments/{ref}.txt
# kind="schema"   → schemas/{ref}.json
#
# D-079 prompt v2s (ADR-005: new files, never in-place edits — the v1
# files stay byte-identical for replay): symmetric widen-or-tighten exit
# guidance, the §3.2 R:R floor stated to the analyzer, the
# adjust_take_profit decision, and the optional trail_distance_pct
# field. v2s compose against v2-suffixed schema files for the same
# reason — editing a shared schema would roll the v1 composed hashes.
_PROMPT_DEFS: dict[str, list[tuple[str, str]]] = {
    "sonnet_filing_analysis_v1": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("fragment", "output_schema"),
        ("schema", "trade_proposal"),
        # S5.3 carry-forward (S5.1 D-2): the no-trade schema is embedded in
        # the prompt body so the LLM knows the NoTradeRecord output shape;
        # editing it must roll the prompt-version hash per ADR-005.
        ("schema", "no_trade_record"),
    ],
    "sonnet_filing_analysis_v2": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("fragment", "output_schema"),
        ("schema", "trade_proposal_v2"),
        ("schema", "no_trade_record"),
    ],
    "opus_proposal_review_v1": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("schema", "opus_proposal_review"),
    ],
    "opus_proposal_review_v2": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("schema", "opus_proposal_review"),
    ],
    # Feature A — time-horizon thesis review on open positions.
    "opus_thesis_review_v1": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("schema", "thesis_review"),
    ],
    "opus_thesis_review_v2": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("schema", "thesis_review_v2"),
    ],
}

# Active prompt per pipeline stage — all stages run v2 (D-079).
#
# loop._compose_decision_id mirrors the analyzer's decision-id
# derivation by importing ACTIVE_SONNET_ANALYSIS_PROMPT, so flipping
# this constant moves both sides together (the v1-era hardcoded
# literal was a silent-mismatch hazard, removed at integration).
ACTIVE_SONNET_ANALYSIS_PROMPT = "sonnet_filing_analysis_v2"
ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT = "opus_proposal_review_v2"
ACTIVE_OPUS_THESIS_REVIEW_PROMPT = "opus_thesis_review_v2"


class PromptBuilder:
    """Loads and composes prompt files into the Anthropic three-block shape.

    Per ADR-005: block 1 (system/role/schema/few-shots — cached, prompt-version
    stable), block 2 (universe/decision context — cached, daily-stable),
    block 3 (filing payload — per-call, not cached). Block 1 and block 2 each
    end with `cache_control: {"type": "ephemeral"}`.
    """

    def __init__(self, prompt_dir: Path) -> None:
        self._dir = prompt_dir

    # -- public API ---------------------------------------------------------

    def prompt_version(self, name: str) -> str:
        """Return composed version id `{name}@{sha256(composed_bytes)[:12]}`.

        The composed bytes are the assembled system block (top-level prompt
        file + included fragments + included schemas). Per ADR-005, modifying
        any included fragment rolls the top-level version (D-031).
        """
        composed = self._composed_system_bytes(name)
        digest = hashlib.sha256(composed).hexdigest()
        return f"{name}@{digest[:_HASH_PREFIX_LEN]}"

    def build_sonnet_filing_analysis(
        self, filing: FilingRow, raw_text: str, ctx: AnalyzerContext
    ) -> ApiRequest:
        return self._build(
            name=ACTIVE_SONNET_ANALYSIS_PROMPT,
            block2_text=self._block2_text(ctx),
            block3_text=self._filing_payload(filing, raw_text),
        )

    def build_opus_proposal_review(
        self, proposal: ProposalRow, source_text_summary: str
    ) -> ApiRequest:
        ctx_summary = (
            f"decision_id={proposal.decision_id}\n"
            f"original_model={proposal.model_id}\n"
            f"original_prompt_version={proposal.prompt_version}\n"
        )
        return self._build(
            name=ACTIVE_OPUS_PROPOSAL_REVIEW_PROMPT,
            block2_text=ctx_summary,
            block3_text=self._proposal_payload(proposal, source_text_summary),
        )

    def build_opus_thesis_review(
        self,
        *,
        proposal: ProposalRow,
        execution_id: int,
        days_held: int,
        current_price: float,
        pct_change_since_entry: float,
        realized_fill_price: float | None,
        realized_dollar_size: float | None,
        time_horizon_days: int,
        stop_loss_price: float,
        take_profit_price: float | None,
        filings_since_entry_summary: str,
        position_zone: int | None = None,
        trail_armed: bool | None = None,
        hwm_gain_pct: float | None = None,
        trigger_reason: str | None = None,
        trigger_detail: str | None = None,
        capacity_pressure_block: str | None = None,
    ) -> ApiRequest:
        """Feature A — compose the Opus thesis-review request for an open
        position whose horizon has elapsed (or an event trigger fired).

        The block-2 context is per-execution (not daily-stable) — Opus's
        cache-hit on these calls is limited to block 1 (role + rules +
        schema). Same trade-off as `opus_proposal_review_v1`.

        The zone/trigger keywords render the "Position state and
        jurisdiction" block the prompt's jurisdiction-zones doctrine
        governs by. They default to None so legacy call sites keep the
        pre-doctrine payload shape byte-identical.

        `capacity_pressure_block`, when supplied (daily-sweep reviews under
        portfolio pressure only), is appended to the END of block 3 (the
        per-call user message) so it never perturbs the cached system
        blocks (block 1 + 2). It carries the weakness ranking and points
        the reviewer at the prompt's own dead-money doctrine + exemptions.
        """
        ctx_summary = (
            f"# Thesis-review context\n"
            f"execution_id: {execution_id}\n"
            f"original_decision_id: {proposal.decision_id}\n"
            f"original_prompt_version: {proposal.prompt_version}\n"
        )
        block3 = (
            f"# Position under review\n"
            f"symbol: {proposal.symbol}\n"
            f"direction: {proposal.direction}\n"
            f"size_pct_requested: {proposal.size_pct_requested}\n"
            f"conviction_at_entry: {proposal.conviction}\n"
            f"original_thesis: {proposal.thesis}\n"
            f"declared_time_horizon_days: {time_horizon_days}\n"
            f"original_stop_loss_price: {stop_loss_price:.2f}\n"
            f"original_take_profit_price: "
            f"{('%.2f' % take_profit_price) if take_profit_price else 'none'}\n"
            f"realized_fill_price: "
            f"{('%.2f' % realized_fill_price) if realized_fill_price else 'unknown'}\n"
            f"realized_dollar_size: "
            f"{('$%.2f' % realized_dollar_size) if realized_dollar_size else 'unknown'}\n"
            f"---\n"
            f"# Current state\n"
            f"days_held_since_entry: {days_held}\n"
            f"current_price: {current_price:.2f}\n"
            f"pct_change_since_entry: {pct_change_since_entry:+.2%}\n"
        )
        if position_zone is not None:
            block3 += self._position_state_block(
                position_zone=position_zone,
                trail_armed=trail_armed,
                hwm_gain_pct=hwm_gain_pct,
                pct_change_since_entry=pct_change_since_entry,
                days_held=days_held,
                trigger_reason=trigger_reason,
                trigger_detail=trigger_detail,
            )
        block3 += (
            f"---\n"
            f"# Filings since entry (for {proposal.symbol})\n"
            f"{filings_since_entry_summary}\n"
        )
        if capacity_pressure_block:
            block3 = f"{block3}---\n{capacity_pressure_block}\n"
        return self._build(
            name=ACTIVE_OPUS_THESIS_REVIEW_PROMPT,
            block2_text=ctx_summary,
            block3_text=block3,
        )

    @staticmethod
    def _position_state_block(
        *,
        position_zone: int,
        trail_armed: bool | None,
        hwm_gain_pct: float | None,
        pct_change_since_entry: float,
        days_held: int,
        trigger_reason: str | None,
        trigger_detail: str | None,
    ) -> str:
        """The structured zone/trigger block every thesis review carries
        (calendar AND event) — the inputs the prompt's jurisdiction-zones
        doctrine governs by, plus the strategy's own base rates."""
        armed = "unknown" if trail_armed is None else str(trail_armed).lower()
        hwm = (
            "unknown"
            if hwm_gain_pct is None
            else f"{hwm_gain_pct:+.2f}%"
        )
        out = (
            f"---\n"
            f"# Position state and jurisdiction\n"
            f"position_zone: {position_zone}\n"
            f"trail_armed: {armed}\n"
            f"hwm_gain_pct: {hwm}\n"
            f"current_gain_pct: {pct_change_since_entry:+.2%}\n"
            f"days_held: {days_held}\n"
            f"review_trigger: {trigger_reason or 'unknown'}\n"
        )
        if trigger_detail:
            out += f"trigger_detail: {trigger_detail}\n"
        out += (
            "---\n"
            "# Base rates (measured on this strategy's own book)\n"
            "- P(reach +50% | reached +30%) ~= 54%\n"
            "- P(double | reached +50%) ~= 38%\n"
            "- Harvest-early policies scored WORST when replayed against "
            "our own book: cutting winners at intermediate gains gave up "
            "more than it protected.\n"
        )
        return out

    # -- internals ----------------------------------------------------------

    def _build(self, *, name: str, block2_text: str, block3_text: str) -> ApiRequest:
        block1_text = self._composed_system_bytes(name).decode("utf-8")
        version = self.prompt_version(name)
        fragment_versions = self._fragment_versions(name)
        system = [
            Block(text=block1_text, cache_control={"type": "ephemeral"}),
            Block(text=block2_text, cache_control={"type": "ephemeral"}),
        ]
        messages = [Message(role="user", content=[Block(text=block3_text)])]
        return ApiRequest(
            system=system,
            messages=messages,
            prompt_version=version,
            fragment_versions=fragment_versions,
        )

    def _composed_system_bytes(self, name: str) -> bytes:
        if name not in _PROMPT_DEFS:
            raise KeyError(f"unknown prompt name: {name}")
        parts: list[bytes] = []
        # Top-level file
        parts.append(self._top_level_path(name).read_bytes())
        # Included fragments + schemas, in declared order
        for kind, ref in _PROMPT_DEFS[name]:
            parts.append(self._dep_path(kind, ref).read_bytes())
        # Use a deterministic separator so concatenation order is unambiguous.
        return b"\n---\n".join(parts)

    def _fragment_versions(self, name: str) -> list[str]:
        out: list[str] = []
        for kind, ref in _PROMPT_DEFS[name]:
            path = self._dep_path(kind, ref)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:_HASH_PREFIX_LEN]
            stem = path.stem
            out.append(f"{stem}@{digest}")
        return out

    def _top_level_path(self, name: str) -> Path:
        return self._dir / f"{name}.txt"

    def _dep_path(self, kind: str, ref: str) -> Path:
        if kind == "fragment":
            return self._dir / "fragments" / f"{ref}.txt"
        if kind == "schema":
            return self._dir / "schemas" / f"{ref}.json"
        raise ValueError(f"unknown dep kind: {kind}")

    def _block2_text(self, ctx: AnalyzerContext) -> str:
        return (
            f"# Decision context\n"
            f"decision_id: {ctx.decision_id}\n"
            f"kill_switch_state: {ctx.kill_switch_state}\n"
            f"open_positions_count: {ctx.open_positions_count}\n"
            f"universe_summary:\n{ctx.universe_summary}\n"
        )

    def _filing_payload(self, filing: FilingRow, raw_text: str) -> str:
        body = self._cap_body(filing, raw_text)
        return (
            f"# Filing under analysis\n"
            f"accession_number: {filing.accession_number}\n"
            f"cik: {filing.cik}\n"
            f"form_type: {filing.form_type}\n"
            f"filed_at: {filing.filed_at.isoformat()}\n"
            f"item_codes: {filing.item_codes or '[]'}\n"
            f"issuer_ticker: {filing.issuer_ticker or ''}\n"
            f"---\n"
            f"{body}\n"
        )

    @staticmethod
    def _cap_body(filing: FilingRow, raw_text: str) -> str:
        original = len(raw_text.encode("utf-8"))
        if original <= _FILING_BODY_BYTE_CAP:
            return raw_text
        # Truncate by bytes — encode, slice, decode with errors='ignore' to
        # avoid splitting a multi-byte sequence mid-character.
        truncated = (
            raw_text.encode("utf-8")[:_FILING_BODY_BYTE_CAP]
            .decode("utf-8", errors="ignore")
        )
        _log.info(
            "filing_body_truncated_at_cap",
            extra={
                "event": "filing_body_truncated_at_cap",
                "filing_id": filing.id,
                "accession": filing.accession_number,
                "original_size": original,
                "cap": _FILING_BODY_BYTE_CAP,
            },
        )
        return truncated + _FILING_BODY_TRUNCATION_MARKER

    def _proposal_payload(self, proposal: ProposalRow, source_text_summary: str) -> str:
        return (
            f"# Proposal under review\n"
            f"symbol: {proposal.symbol}\n"
            f"direction: {proposal.direction}\n"
            f"size_pct_requested: {proposal.size_pct_requested}\n"
            f"conviction: {proposal.conviction}\n"
            f"thesis: {proposal.thesis}\n"
            f"---\n"
            f"# Source filing summary\n"
            f"{source_text_summary}\n"
            f"---\n"
            f"# Original raw response\n"
            f"{proposal.raw_response}\n"
        )
