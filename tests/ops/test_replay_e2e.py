"""End-to-end test of the replay CLI (`ops/replay/run_replay.py`).

Builds a synthetic journal (real schema via journal.migrate) and a
pre-populated offline bars cache, runs main() with --offline, and checks the
produced CSVs and markdown report. Doubles as the sample-output generator
for the task #5 completion report.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest
from ops.replay.run_replay import main

from journal.migrate import apply_migrations
from journal.models import FilingRow, PromptRow, ProposalRow
from journal.repo import insert_filing, insert_prompt, insert_proposal


def _seed_proposal(
    db: str,
    n: int,
    symbol: str,
    created_at: str,
    stop: float,
    tp: float | None,
    horizon: int,
) -> None:
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"0000000000-26-{n:06d}",
            cik=1000 + n,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 1, tzinfo=dt.UTC),
            fetched_at=dt.datetime(2026, 4, 1, tzinfo=dt.UTC),
            raw_text_path=f"/tmp/f{n}.txt",
            content_hash=f"hash{n}",
        ),
    )
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{n}",
            model_id="claude-opus-4-7",
            prompt_version="v1",
            raw_response=json.dumps(
                {
                    "symbol": symbol,
                    "stop_loss_price": stop,
                    "take_profit_price": tp,
                    "time_horizon_days": horizon,
                }
            ),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            conviction=7,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    conn = sqlite3.connect(db)
    conn.execute("UPDATE proposals SET created_at = ? WHERE id = ?", (created_at, pid))
    conn.commit()
    conn.close()


def _cache_doc(symbol: str, bars: list[tuple[str, float, float, float, float]]) -> dict:
    return {
        "version": 1,
        "symbol": symbol,
        "fetched_at": "2026-06-09T00:00:00+00:00",
        "start": "2026-04-01",
        "end": "2026-12-31",
        "feed": "sip",
        "bars": [
            {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 10000}
            for d, o, h, lo, c in bars
        ],
        "splits": [],
    }


@pytest.fixture
def env(tmp_path: Path) -> dict:
    db = str(tmp_path / "journal.db")
    apply_migrations(db)
    insert_prompt(
        db, PromptRow(prompt_version="v1", name="analyzer", file_path="x.md", content_hash="h")
    )
    # WINR: pre-regime-split proposal (Mon 2026-05-04 pre-open), TP hit day 2.
    _seed_proposal(db, 1, "WINR", "2026-05-04 12:00:00", stop=8.0, tp=11.0, horizon=10)
    # LOSR: post-regime-split proposal (Mon 2026-05-11 pre-open), stop hit day 2.
    _seed_proposal(db, 2, "LOSR", "2026-05-11 12:00:00", stop=18.0, tp=30.0, horizon=10)
    # DRFT: drifts sideways into the time-stop; trailing run differs from TP run.
    _seed_proposal(db, 3, "DRFT", "2026-05-11 12:00:00", stop=4.0, tp=9.0, horizon=4)

    cache = tmp_path / "bars"
    cache.mkdir()
    (cache / "WINR.json").write_text(
        json.dumps(
            _cache_doc(
                "WINR",
                [
                    ("2026-05-04", 10.0, 10.4, 9.8, 10.2),
                    ("2026-05-05", 10.3, 11.5, 10.1, 11.2),
                ],
            )
        )
    )
    (cache / "LOSR.json").write_text(
        json.dumps(
            _cache_doc(
                "LOSR",
                [
                    ("2026-05-11", 20.0, 20.5, 19.5, 20.0),
                    ("2026-05-12", 19.0, 19.2, 17.5, 17.8),
                ],
            )
        )
    )
    (cache / "DRFT.json").write_text(
        json.dumps(
            _cache_doc(
                "DRFT",
                [
                    ("2026-05-11", 5.0, 5.2, 4.9, 5.1),
                    ("2026-05-12", 5.1, 5.3, 5.0, 5.2),
                    ("2026-05-13", 5.2, 5.4, 5.1, 5.3),
                    ("2026-05-14", 5.3, 5.5, 5.2, 5.4),
                    ("2026-05-15", 5.4, 5.6, 5.3, 5.5),
                ],
            )
        )
    )
    return {"db": db, "cache": str(cache), "out": str(tmp_path / "out")}


def test_cli_end_to_end_offline(env, capsys):
    rc = main(
        [
            "--journal",
            env["db"],
            "--out",
            env["out"],
            "--cache",
            env["cache"],
            "--offline",
        ]
    )
    assert rc == 0
    out_dir = Path(env["out"])
    for name in (
        "report.md",
        "trades_proposal_tp.csv",
        "trades_trailing.csv",
        "equity_proposal_tp.csv",
        "equity_trailing.csv",
    ):
        assert (out_dir / name).exists(), name

    with (out_dir / "trades_proposal_tp.csv").open() as f:
        rows = {r["symbol"]: r for r in csv.DictReader(f)}
    assert rows["WINR"]["exit_reason"] == "take_profit"
    assert float(rows["WINR"]["pnl_usd"]) == pytest.approx(100.0, abs=0.01)
    assert rows["LOSR"]["exit_reason"] == "stop"
    assert float(rows["LOSR"]["pnl_usd"]) == pytest.approx(-100.0, abs=0.01)
    assert rows["DRFT"]["exit_reason"] == "time_stop"
    assert rows["DRFT"]["status"] == "closed"

    report = (out_dir / "report.md").read_text()
    assert "## Simulation assumptions" in report
    assert "pre 2026-05-08" in report and "post 2026-05-08" in report
    assert "TP-capping cost" in report
    # trailing mode must not exit WINR at the TP
    with (out_dir / "trades_trailing.csv").open() as f:
        trows = {r["symbol"]: r for r in csv.DictReader(f)}
    assert trows["WINR"]["exit_reason"] != "take_profit"

    stdout = capsys.readouterr().out
    assert "Loaded 3 replayable proposals" in stdout


def test_offline_cache_miss_is_reported_not_fatal(env, tmp_path):
    Path(env["cache"], "LOSR.json").unlink()
    rc = main(
        ["--journal", env["db"], "--out", env["out"], "--cache", env["cache"], "--offline"]
    )
    assert rc == 0
    report = Path(env["out"], "report.md").read_text()
    assert "bars unavailable" in report
    assert "LOSR" in report


def test_offline_cache_from_other_feed_not_served(env):
    # A cache written from --feed iex must not silently serve a sip run.
    path = Path(env["cache"], "WINR.json")
    doc = json.loads(path.read_text())
    doc["feed"] = "iex"
    path.write_text(json.dumps(doc))
    rc = main(
        ["--journal", env["db"], "--out", env["out"], "--cache", env["cache"], "--offline"]
    )
    assert rc == 0
    report = Path(env["out"], "report.md").read_text()
    assert "bars unavailable" in report and "WINR" in report


def test_default_horizon_clamped_to_fetch_window(env, capsys):
    rc = main(
        [
            "--journal",
            env["db"],
            "--out",
            env["out"],
            "--cache",
            env["cache"],
            "--offline",
            "--default-horizon",
            "200",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "clamped to 75" in err


def test_journal_not_mutated(env):
    before = Path(env["db"]).read_bytes()
    main(["--journal", env["db"], "--out", env["out"], "--cache", env["cache"], "--offline"])
    assert Path(env["db"]).read_bytes() == before
