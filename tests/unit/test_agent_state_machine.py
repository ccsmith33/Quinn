"""S5.6 AC-3 — agent-loop state machine.

Pure unit tests over the enum + transition table. No asyncio.
"""

from __future__ import annotations

import pytest


def test_state_values_match_story() -> None:
    """AC-3: AgentState carries exactly the five states from the story
    diagram, with stable string values used in structured logs."""
    from app.state import AgentState

    assert AgentState.BOOTING.value == "booting"
    assert AgentState.IDLE.value == "idle"
    assert AgentState.PROCESSING.value == "processing"
    assert AgentState.SHUTTING_DOWN.value == "shutting_down"
    assert AgentState.STOPPED.value == "stopped"
    assert {s.value for s in AgentState} == {
        "booting",
        "idle",
        "processing",
        "shutting_down",
        "stopped",
    }


def test_legal_transitions_round_trip() -> None:
    """AC-3: the transition table accepts the legal moves drawn in the
    story's state diagram."""
    from app.state import AgentState, Transitions

    t = Transitions()
    # Boot to IDLE.
    assert t.can(AgentState.BOOTING, AgentState.IDLE)
    # IDLE → PROCESSING (new filing) and PROCESSING → IDLE (success).
    assert t.can(AgentState.IDLE, AgentState.PROCESSING)
    assert t.can(AgentState.PROCESSING, AgentState.IDLE)
    # IDLE → SHUTTING_DOWN (SIGTERM at idle).
    assert t.can(AgentState.IDLE, AgentState.SHUTTING_DOWN)
    # PROCESSING → SHUTTING_DOWN (SIGTERM mid-flight; story rule says the
    # in-flight filing finishes first, but the transition is legal once
    # the current pipeline returns).
    assert t.can(AgentState.PROCESSING, AgentState.SHUTTING_DOWN)
    # SHUTTING_DOWN → STOPPED.
    assert t.can(AgentState.SHUTTING_DOWN, AgentState.STOPPED)


def test_illegal_transitions_rejected() -> None:
    """AC-3: an illegal transition raises IllegalAgentTransition."""
    from app.state import AgentState, IllegalAgentTransition, Transitions

    t = Transitions()
    # STOPPED is terminal.
    assert not t.can(AgentState.STOPPED, AgentState.IDLE)
    assert not t.can(AgentState.STOPPED, AgentState.BOOTING)
    # No skipping over IDLE.
    assert not t.can(AgentState.BOOTING, AgentState.PROCESSING)
    assert not t.can(AgentState.BOOTING, AgentState.STOPPED)
    # Can't run a new filing while shutting down.
    assert not t.can(AgentState.SHUTTING_DOWN, AgentState.PROCESSING)

    with pytest.raises(IllegalAgentTransition):
        t.assert_legal(AgentState.STOPPED, AgentState.IDLE)


def test_self_transitions_rejected() -> None:
    """AC-3: re-entering the current state is not a legal transition.

    The state machine only encodes deltas; staying in the same state means
    no transition fired and the table should reject re-entry to make
    misuse loud.
    """
    from app.state import AgentState, Transitions

    t = Transitions()
    for s in AgentState:
        assert not t.can(s, s)
