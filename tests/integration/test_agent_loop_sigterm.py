"""S5.6 AC-10 — SIGTERM-during-processing test (load-bearing).

The test fires SIGTERM while a filing is mid-analyze. The in-flight
filing must complete; the next filing in the queue must NOT be
consumed; the loop must exit 0 within the grace window.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from app.loop import AgentLoop
from journal.models import FilingRow
from journal.repo import connect


@pytest.mark.asyncio
async def test_agent_loop_sigterm_completes_in_flight(
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    from app.loop import AgentLoop
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    f1 = make_filing(accession="0001234567-26-000010", cik=320193)
    f2 = make_filing(accession="0001234567-26-000011", cik=320193)
    f3 = make_filing(accession="0001234567-26-000012", cik=320193)

    # Pre-seed prefilter rows so the BOOTING crash-recovery scan does
    # not also re-queue these filings (the test wants strict control
    # over consumption ordering — see queue.put calls below). The
    # decision "accept" via material_8k_bypass keeps the analyzer path
    # live; that's what we want to delay-instrument with f2.
    from journal.models import PrefilterDecisionRow
    from journal.repo import insert_prefilter_decision
    for f in (f1, f2, f3):
        insert_prefilter_decision(
            db_path,
            PrefilterDecisionRow(
                filing_id=f.id,
                decision="accept",
                rule_fired="material_8k_bypass",
                reason_detail='item_codes=["1.01"]',
            ),
        )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f1)
    await queue.put(f2)
    await queue.put(f3)

    # Two analyze-purpose responses (we'll only consume two filings) +
    # two opus reviews.
    fake_anthropic.responses_by_purpose = {
        "analyze": [
            _valid_proposal_json(conviction=9),
            _valid_proposal_json(conviction=9),
        ],
        "review": [_opus_ratify_json(), _opus_ratify_json()],
    }
    # Make the SECOND filing's analyze call sleep so SIGTERM lands during
    # the second pipeline. The decision_id is computed deterministically;
    # we precompute it for f2 to set the delay.
    from analyzer.sonnet import compute_decision_id
    from prompts.loader import PromptBuilder
    from pathlib import Path

    from prompts.loader import ACTIVE_SONNET_ANALYSIS_PROMPT

    pb = PromptBuilder(Path("src/prompts"))
    pv = pb.prompt_version(ACTIVE_SONNET_ANALYSIS_PROMPT)
    f2_decision_id = compute_decision_id(
        filing_id=f2.id, model_id="claude-sonnet-4-6", prompt_version=pv
    )
    fake_anthropic.delay_seconds_by_decision_id[f2_decision_id] = 1.5

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())

    # Wait until the loop is mid-processing the second filing, then SIGTERM.
    await asyncio.sleep(0.4)  # f1 finishes fast; we want to be inside f2's sleep
    os.kill(os.getpid(), signal.SIGTERM)

    rc = await asyncio.wait_for(runner, timeout=10.0)
    assert rc == 0

    # Journal must reflect: f1 + f2 both went through analyzer +
    # execution; f3 must NOT have been consumed (prefilter rows for all
    # three were pre-seeded to disable the recovery scan).
    with connect(db_path) as conn:
        proposals_count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        executions_count = conn.execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        executed_filing_ids = {
            r["filing_id"] for r in conn.execute(
                "SELECT pr.filing_id FROM proposals pr "
                "JOIN executions e ON e.proposal_id = pr.id"
            ).fetchall()
        }

    assert proposals_count == 2, (
        f"only f1 and f2 should have proposals (f3 not consumed); got {proposals_count}"
    )
    assert executions_count == 2, (
        "both fully-processed proposals must have execution rows — no orphan proposal"
    )
    assert f3.id not in executed_filing_ids
    # f3 is still in the queue, followed by the None sentinel the SIGTERM
    # handler enqueued (request_shutdown) to wake an idle consumer. The
    # consumer exited on the flag check after f2's pipeline, so neither
    # was consumed.
    remaining = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    assert len(remaining) == 2
    assert remaining[0] is not None and remaining[0].id == f3.id
    assert remaining[1] is None


async def _await_completion(runner: asyncio.Task, timeout: float = 5.0) -> int:
    """Await `runner` WITHOUT cancelling it on timeout.

    `asyncio.wait_for` would cancel the task at timeout; `AgentLoop.run`
    catches CancelledError at `await consumer` and still tears down
    gracefully with rc=0 — exactly the false-pass that would hide the
    original never-wakes bug. `asyncio.wait` leaves a hung runner hung,
    so the timeout is observable; we then cancel only to keep the event
    loop clean before failing the test.
    """
    done, pending = await asyncio.wait({runner}, timeout=timeout)
    if pending:
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — already failing; keep the loop clean
            pass
        pytest.fail(
            f"AgentLoop.run() did not complete within {timeout}s of the "
            "shutdown request — the idle consumer was never woken"
        )
    return runner.result()


class _StubRssLoop:
    """RSS discovery stand-in exposing the surface `AgentLoop` touches:
    `start()` (sets `_task`), `stop()`, and the `_task` attribute."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_sigterm_wakes_idle_consumer_and_runs_shutdown(
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
) -> None:
    """The restart-hang fix (load-bearing): SIGTERM while the consumer is
    IDLE — blocked on an EMPTY `ingestion_queue.get()` — must complete
    `run()` promptly via the handler's sentinel wake, and `_shutdown()`
    must actually run (RSS stop + reconciler stop invoked). Pre-fix, the
    handler only set the flag: nobody woke `queue.get()`, so the process
    hung until systemd's SIGKILL and teardown never executed.
    """
    import dataclasses

    from app.state import AgentState

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    stub_rss = _StubRssLoop()
    discovered: asyncio.Queue = asyncio.Queue()
    components = dataclasses.replace(
        build_components(queue=queue),
        rss_discovery_loop=stub_rss,
        detail_fetcher=object(),  # pump never dequeues an item before cancel
        discovered_queue=discovered,
    )
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())

    # Wait until boot finished and the consumer is parked on the empty queue.
    for _ in range(200):
        if loop.state is AgentState.IDLE:
            break
        await asyncio.sleep(0.01)
    assert loop.state is AgentState.IDLE
    await asyncio.sleep(0.05)

    os.kill(os.getpid(), signal.SIGTERM)

    # Pre-fix the consumer stayed blocked on queue.get() and run() never
    # returned. NOTE: `asyncio.wait` (not `wait_for`) is load-bearing —
    # `wait_for`'s timeout path CANCELS the runner, and `run()` swallows
    # CancelledError at `await consumer` and proceeds to `_shutdown()`,
    # which would mask the hang as a clean rc=0.
    rc = await _await_completion(runner)
    assert rc == 0
    assert loop.shutdown_requested is True
    assert loop.state is AgentState.STOPPED
    # _shutdown() ran its teardown: RSS discovery stopped (cursor persist
    # lives inside stop() in prod) and the reconciler stopped.
    assert stub_rss.stopped is True
    assert components.reconciler.stopped is True


@pytest.mark.asyncio
async def test_request_shutdown_wakes_idle_consumer_directly(
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
) -> None:
    """`AgentLoop.request_shutdown()` alone (no OS signal) wakes an idle
    consumer: flag set + None sentinel on the ingestion queue."""
    from app.state import AgentState

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())
    for _ in range(200):
        if loop.state is AgentState.IDLE:
            break
        await asyncio.sleep(0.01)
    assert loop.state is AgentState.IDLE

    loop.request_shutdown()

    rc = await _await_completion(runner)
    assert rc == 0
    assert components.reconciler.stopped is True


@pytest.mark.asyncio
async def test_double_request_shutdown_is_harmless(
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
) -> None:
    """Two sentinels (e.g. SIGTERM followed by SIGINT, or a repeated
    SIGTERM from systemd) must not wedge or error: the consumer exits on
    the first None; the second stays inert in the queue."""
    from app.state import AgentState

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())
    for _ in range(200):
        if loop.state is AgentState.IDLE:
            break
        await asyncio.sleep(0.01)

    loop.request_shutdown()
    loop.request_shutdown()  # double sentinel while already shutting down

    rc = await _await_completion(runner)
    assert rc == 0
    assert loop.state is AgentState.STOPPED
    # Exactly one leftover sentinel — consumed nothing else, raised nothing.
    assert queue.qsize() == 1
    assert queue.get_nowait() is None
