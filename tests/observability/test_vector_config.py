"""S8.1 — Vector config validation (AC-4, AC-5).

The Vector config at `ops/vector/quinn.toml` ships journald entries to
Better Stack via the configured token, with disk-buffered replay so the
application code never blocks on delivery.

We don't shell out to `vector validate` here (Vector may not be on the
path in dev). Instead we parse the TOML and assert the structural
invariants the story requires.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parents[2] / "ops" / "vector" / "quinn.toml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    assert CONFIG_PATH.is_file(), f"missing Vector config at {CONFIG_PATH}"
    with CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_config_has_journald_source(cfg: dict) -> None:
    sources = cfg.get("sources", {})
    journald = next((s for s in sources.values() if s.get("type") == "journald"), None)
    assert journald is not None, "expected a journald source"
    units = journald.get("include_units") or journald.get("units") or []
    # Quinn services are quinn-agent, quinn-bot, quinn-http, quinn-universe
    # — at least one must be referenced (S8.3 adds the unit files).
    assert any("quinn" in str(u) for u in units), "journald source should include quinn-* units"


def test_config_ships_to_better_stack(cfg: dict) -> None:
    sinks = cfg.get("sinks", {})
    assert sinks, "expected at least one sink"
    bs = next(
        (
            s
            for s in sinks.values()
            if "betterstack" in s.get("type", "").lower()
            or "logtail" in s.get("type", "").lower()
            or "http" == s.get("type")
        ),
        None,
    )
    assert bs is not None, "expected a Better Stack / Logtail / http sink"
    # Auth must come from env var, never embedded in the file
    serialized = repr(bs)
    assert "${LOG_SINK_TOKEN" in serialized or "$LOG_SINK_TOKEN" in serialized, (
        "Better Stack token must be sourced from $LOG_SINK_TOKEN env var"
    )


def test_config_has_disk_buffer_for_replay(cfg: dict) -> None:
    """AC-5: Vector buffers and replays when the sink returns."""
    sinks = cfg.get("sinks", {})
    sink_with_buffer = next((s for s in sinks.values() if isinstance(s.get("buffer"), dict)), None)
    assert sink_with_buffer is not None, "at least one sink must declare a buffer"
    buf = sink_with_buffer["buffer"]
    assert buf.get("type") == "disk", "buffer must be disk-backed for replay across restarts"
    assert buf.get("max_size", 0) > 0, "disk buffer must have a positive max_size"


def test_config_does_not_embed_secret_token(cfg: dict) -> None:
    """The on-disk config must never carry the raw token."""
    text = CONFIG_PATH.read_text()
    # placeholder pattern is fine; raw token (anything that looks like a
    # 32+ char hex/alnum without env-var syntax) is not.
    assert "LOG_SINK_TOKEN" in text  # env var reference present
    # crude check: no `Authorization = "Bearer abcdef...long..."` literal
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "Bearer " in stripped:
            assert "${" in stripped or "$LOG_SINK_TOKEN" in stripped, (
                f"raw bearer token literal in config: {stripped}"
            )
