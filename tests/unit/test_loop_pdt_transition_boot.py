# PDT-TRANSITION-D-077: boot-seam ordering tests (ADR-012 §4.2 step 5).
# Survives P2 with the converter; removed in the post-soak cleanup.
"""`AgentLoop._boot` must run the transition converter BEFORE the
deferred replayer — the converter's drain invalidates `deferred_sells`
rows the replayer would otherwise double-sell against freshly-converted
GTC orders. Ordering is load-bearing; these tests pin it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.composition import AgentComponents
from app.loop import AgentLoop
from journal.migrate import apply_migrations
from journal.repo import JournalRepo


class _StubReconciler:
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class _Recorder:
    def __init__(self, calls: list[str], name: str, fail: bool = False) -> None:
        self._calls = calls
        self._name = name
        self._fail = fail

    def run(self) -> None:
        self._calls.append(self._name)
        if self._fail:
            raise RuntimeError(f"{self._name} exploded")


def _components(
    db_path: str,
    *,
    converter: Any | None,
    replayer: Any | None,
) -> AgentComponents:
    none: Any = None
    return AgentComponents(
        universe=none,
        prefilter=none,
        analyzer=none,
        reviewer=none,
        proposal_store=none,
        validator=none,
        sizer=none,
        submitter=none,
        execution=None,
        broker=none,
        reconciler=_StubReconciler(),
        killswitch=none,
        ingestion_queue=asyncio.Queue(),
        journal=JournalRepo(db_path),
        deferred_replayer=replayer,
        pdt_transition_converter=converter,
    )


def _boot(components: AgentComponents) -> None:
    loop = AgentLoop(components=components)
    asyncio.run(loop._boot())


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "journal.db")
    apply_migrations(path)
    return path


def test_boot_runs_converter_before_replayer(db_path: str) -> None:
    calls: list[str] = []
    components = _components(
        db_path,
        converter=_Recorder(calls, "converter"),
        replayer=_Recorder(calls, "replayer"),
    )
    _boot(components)
    assert calls == ["converter", "replayer"]


def test_boot_converter_error_does_not_block_boot_or_replayer(
    db_path: str,
) -> None:
    """Converter failures are logged, never propagated — boot completes
    and the replayer still runs (its own supersede guard protects it
    when the drain didn't happen)."""
    calls: list[str] = []
    components = _components(
        db_path,
        converter=_Recorder(calls, "converter", fail=True),
        replayer=_Recorder(calls, "replayer"),
    )
    _boot(components)
    assert calls == ["converter", "replayer"]


def test_boot_without_converter_still_runs_replayer(db_path: str) -> None:
    """Legacy/test component bundles (converter=None) boot unchanged."""
    calls: list[str] = []
    components = _components(
        db_path,
        converter=None,
        replayer=_Recorder(calls, "replayer"),
    )
    _boot(components)
    assert calls == ["replayer"]
