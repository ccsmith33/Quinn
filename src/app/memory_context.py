"""Shared rail for Quinn's LLM memory layer.

A `MemoryContextAssembler` holds an ordered registry of *provider*
callables. Each provider inspects a `MemoryQuery` (what the LLM is about
to be asked, and about which symbol) and either returns a `MemorySection`
to inject or `None` to stay silent. `assemble()` concatenates the
returned sections, in REGISTERED order, into one deterministic block that
a caller appends to the per-call LLM context.

The four real providers (doctrine, symbol_history, desk_journal,
calibration) are built separately and register themselves here; this
module owns only the contract + the assembly rail, and ships with NO
real provider registered.

Invariants:
- Deterministic: identical inputs → byte-identical output. No timestamps,
  no clock, no dict ordering — sections render in registration order.
- Fail-open: a provider that raises is caught, logged
  (`memory.provider_failed`), and skipped. Memory is an enrichment; it
  must NEVER break a trading call. `assemble()` returns `None` when no
  provider contributed, and callers treat `None` as "no memory block".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from observability.log_port import get_logger

log = get_logger(__name__)


MemoryPurpose = Literal["analyze", "proposal_review", "thesis_review"]


@dataclass(frozen=True)
class MemoryQuery:
    """What the LLM is about to be asked — the input every provider sees.

    `symbol` is None on pre-symbol calls (a filing whose issuer ticker did
    not resolve). `execution_id` / `conviction` are populated only where
    the call site has them (thesis review has both; filing analysis has
    neither yet).
    """

    symbol: str | None
    purpose: MemoryPurpose
    execution_id: int | None = None
    conviction: int | None = None


@dataclass(frozen=True)
class MemorySection:
    """One provider's contribution. `title` heads the rendered block;
    `provider_name` identifies the source for logging/telemetry."""

    title: str
    body: str
    provider_name: str


# Provider contract: `provide(ctx: MemoryQuery) -> MemorySection | None`.
Provider = Callable[[MemoryQuery], MemorySection | None]


class MemoryContextAssembler:
    """Ordered registry of memory providers + the assembly rail."""

    def __init__(self) -> None:
        self._providers: list[tuple[str, Provider]] = []

    def register(self, name: str, provider: Provider) -> None:
        """Append a provider under `name`. Registration order IS render
        order — the assembled block is byte-stable for a fixed registry."""
        self._providers.append((name, provider))

    def assemble(self, query: MemoryQuery) -> str | None:
        """Run every provider in registration order and concatenate the
        sections each returns. Returns None when nothing contributed.

        A provider that raises is logged and skipped — one bad provider
        never denies the others or breaks the caller's trading path.
        """
        sections: list[MemorySection] = []
        for name, provider in self._providers:
            try:
                section = provider(query)
            except Exception as e:  # noqa: BLE001 — fail-open by design
                log.warning(
                    "memory.provider_failed",
                    extra={
                        "event": "memory.provider_failed",
                        "provider": name,
                        "error": str(e),
                    },
                )
                continue
            if section is not None:
                sections.append(section)
        if not sections:
            return None
        return "\n\n".join(
            f"## MEMORY: {s.title}\n{s.body}" for s in sections
        )


__all__ = [
    "MemoryContextAssembler",
    "MemoryPurpose",
    "MemoryQuery",
    "MemorySection",
    "Provider",
]
