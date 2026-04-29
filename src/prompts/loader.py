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

_HASH_PREFIX_LEN = 12


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
    "opus_proposal_review_v1": [
        ("fragment", "role"),
        ("fragment", "rules_invariant"),
        ("schema", "opus_proposal_review"),
    ],
}


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
            name="sonnet_filing_analysis_v1",
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
            name="opus_proposal_review_v1",
            block2_text=ctx_summary,
            block3_text=self._proposal_payload(proposal, source_text_summary),
        )

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
        return (
            f"# Filing under analysis\n"
            f"accession_number: {filing.accession_number}\n"
            f"cik: {filing.cik}\n"
            f"form_type: {filing.form_type}\n"
            f"filed_at: {filing.filed_at.isoformat()}\n"
            f"item_codes: {filing.item_codes or '[]'}\n"
            f"issuer_ticker: {filing.issuer_ticker or ''}\n"
            f"---\n"
            f"{raw_text}\n"
        )

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
