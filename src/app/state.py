"""S5.6 AC-3 — typed state machine for the agent loop.

The agent loop is a single asyncio task. Its in-memory state machine
guards transitions inside the running process; the journal is the
durable source of truth for everything else. Per the story diagram:

    BOOTING -> IDLE -> PROCESSING -> IDLE
                  \\                    /
                   \\-> SHUTTING_DOWN -<-
                              |
                              v
                          STOPPED

HALTED is NOT a state-machine state — it's a flag (kill-switch state)
checked at the execution boundary. See `app.loop` for the runtime
glue; this module is pure types so it stays cheap to test.

Per D-052 (no orchestration library): keeping the table as a frozen set
of (from, to) tuples is correct at v1's transition count (~5). Avoid
introducing `transitions`, `automat`, or peer libraries; reviewers
flagged "negative-value dependency" risk.
"""

from __future__ import annotations

from enum import Enum


class AgentState(Enum):
    """Five-state lifecycle for the agent loop.

    Values are the JSON-stable strings emitted by AC-15's structured logs
    and consumed by the bot's `/status` surface (FR-34, S7.2).
    """

    BOOTING = "booting"
    IDLE = "idle"
    PROCESSING = "processing"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class IllegalAgentTransition(Exception):
    """Raised when a caller attempts a transition not in the table."""

    def __init__(self, src: AgentState, dst: AgentState) -> None:
        super().__init__(f"illegal transition {src.value} -> {dst.value}")
        self.src = src
        self.dst = dst


# Legal moves drawn straight off the story's state diagram.
_LEGAL: frozenset[tuple[AgentState, AgentState]] = frozenset(
    {
        (AgentState.BOOTING, AgentState.IDLE),
        (AgentState.IDLE, AgentState.PROCESSING),
        (AgentState.PROCESSING, AgentState.IDLE),
        (AgentState.IDLE, AgentState.SHUTTING_DOWN),
        (AgentState.PROCESSING, AgentState.SHUTTING_DOWN),
        (AgentState.SHUTTING_DOWN, AgentState.STOPPED),
        # BOOTING -> SHUTTING_DOWN handles SIGTERM during boot before the
        # consumer task is started. The story focuses on post-boot moves
        # but the boot path needs an exit too.
        (AgentState.BOOTING, AgentState.SHUTTING_DOWN),
    }
)


class Transitions:
    """Pure transition guard. Stateless; safe to share across instances."""

    def can(self, src: AgentState, dst: AgentState) -> bool:
        return (src, dst) in _LEGAL

    def assert_legal(self, src: AgentState, dst: AgentState) -> None:
        if not self.can(src, dst):
            raise IllegalAgentTransition(src, dst)


__all__ = [
    "AgentState",
    "IllegalAgentTransition",
    "Transitions",
]
