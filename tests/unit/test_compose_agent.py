"""S5.6 AC-2 — composition unit tests.

Verifies `compose_agent` constructs every dependency from `Config` and
returns a fully-wired `AgentLoop`. Single-code-path invariant (D-007 /
AC-13) is enforced by `test_single_code_path_lint` (lint test) — this
file confirms construction-time behavior.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from broker.alpaca import LIVE_ENDPOINT, PAPER_ENDPOINT
from config.loader import (
    AnalyzerConfig,
    AppConfig,
    ExecutionConfig,
    IngestionConfig,
    KillSwitchConfig,
    ObservabilityConfig,
    PrefilterConfig,
    ReconcilerConfig,
)
from config.secrets import Secrets
from journal.migrate import apply_migrations
from journal.models import UniverseMemberRow, UniverseSnapshotRow
from journal.repo import (
    JournalRepo,
    insert_universe_member,
    insert_universe_snapshot,
)


def _config(
    *, broker_mode: str = "paper", tmp_path: Path | None = None
) -> AppConfig:
    if tmp_path is not None:
        cursor_path = str(tmp_path / "rss_cursor.json")
        raw_root = str(tmp_path / "raw")
    else:
        cursor_path = "/var/lib/quinn/state/rss_cursor.json"
        raw_root = "/var/lib/quinn/raw"
    return AppConfig(
        ingestion=IngestionConfig(
            rss_poll_seconds_market=60,
            rss_poll_seconds_offhours=300,
            reconciler_interval_seconds=21600,
            edgar_user_agent="Quinn-Test/v1 test@example.com",
            rss_cursor_path=cursor_path,
            raw_filings_root=raw_root,
        ),
        prefilter=PrefilterConfig(similarity_threshold=0.97, minhash_perms=128),
        analyzer=AnalyzerConfig(
            sonnet_model_id="claude-sonnet-4-6",
            opus_model_id="claude-opus-4-7",
            opus_review_conviction_threshold=7,
            sonnet_max_output_tokens=4096,
        ),
        execution=ExecutionConfig(
            broker_mode=broker_mode,  # type: ignore[arg-type]
            ks4_pct_cap=0.20,
            ks4_absolute_cap_usd=1000.0,
            ks5_max_concurrent=5,
            ks7_cash_reserve_pct=0.05,
            sizing_mid_pct=0.05,
            sizing_high_pct=0.10,
        ),
        reconciler=ReconcilerConfig(interval_seconds_market=300),
        killswitch=KillSwitchConfig(
            ks1_daily_loss_pct=0.03,
            ks2_trailing_dd_pct=0.10,
            ks3_consecutive_losses=6,
        ),
        observability=ObservabilityConfig(
            log_level="INFO", betterstack_endpoint="https://example"
        ),
    )


def _secrets(*, paper: bool = True) -> Secrets:
    endpoint = PAPER_ENDPOINT if paper else LIVE_ENDPOINT
    return Secrets(
        anthropic_api_key=SecretStr("sk-ant-test"),
        alpaca_api_key_id=SecretStr("AKTEST"),
        alpaca_api_secret_key=SecretStr("secret"),
        alpaca_endpoint=SecretStr(endpoint),
        telegram_bot_token=SecretStr("1234:abc"),
        telegram_operator_chat_id=SecretStr("999"),
        kill_switch_hmac_key=SecretStr("hmac-32-bytes-test-key-padding"),
        backup_b2_key_id=SecretStr("b2id"),
        backup_b2_application_key=SecretStr("b2key"),
        backup_b2_bucket=SecretStr("bkt"),
        log_sink_token=SecretStr("logtok"),
        dashboard_user=SecretStr("operator"),
        dashboard_password=SecretStr("dashboard-test-pass"),
    )


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    # Universe must have a snapshot or load_latest raises.
    snap_id = insert_universe_snapshot(
        str(p),
        UniverseSnapshotRow(
            snapshot_date=dt.date(2026, 4, 28),
            sec_tickers_hash="x" * 64,
            alpaca_assets_hash="y" * 64,
            yfinance_failures=0,
            member_count=1,
            is_degraded=0,
        ),
    )
    insert_universe_member(
        str(p),
        UniverseMemberRow(
            snapshot_id=snap_id,
            cik=320193,
            ticker="AAPL",
            exchange="NASDAQ",
            market_cap=3e12,
            prev_close=170.0,
        ),
    )
    return str(p)


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Minimal prompts-dir fixture so PromptBuilder can compose."""
    src_prompts = Path("src/prompts")
    return src_prompts


def test_compose_agent_constructs_all_components(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """AC-2: compose_agent returns an AgentLoop with every AgentComponents
    field populated. We patch the Alpaca SDK init so this stays a unit
    test (no live network)."""
    from app.composition import compose_agent

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)

    # Patch the SDK constructors so the unit test doesn't require live
    # creds. The seam is at construction time; runtime methods are not
    # invoked here.
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
    ):
        loop = compose_agent(
            cfg,
            secrets=secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    c = loop.components
    assert c.universe is not None
    assert c.prefilter is not None
    assert c.analyzer is not None
    assert c.reviewer is not None
    assert c.proposal_store is not None
    # The story's dev-notes list `execution: ExecutionLayer` as a single
    # field; the implementation surfaces the actual v1 components
    # individually (validator + sizer + submitter), with `execution`
    # kept as a placeholder for forward shape parity.
    assert c.validator is not None
    assert c.sizer is not None
    assert c.submitter is not None
    assert c.broker is not None
    assert c.reconciler is not None
    assert c.killswitch is not None
    assert c.ingestion_queue is not None
    assert c.journal is journal


def test_compose_agent_wires_telegram_alerter_into_reconciler(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """S5.6 carry-fwd S6.5 reviewer M-3 (HIGH): the reconciler must be
    constructed with a non-None alerter so position-discrepancy
    notifications reach Telegram, not just structured logs."""
    from app.composition import compose_agent

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
    ):
        loop = compose_agent(
            cfg,
            secrets=secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    rec = loop.components.reconciler
    assert rec._alerter is not None, (  # type: ignore[attr-defined]
        "Reconciler must be wired with the Telegram alerter "
        "(S6.5 reviewer M-3)"
    )
    # The adapter exposes `notify(message)` to satisfy S6.5's
    # structural Protocol.
    assert callable(getattr(rec._alerter, "notify", None))  # type: ignore[attr-defined]


def test_compose_agent_wires_alert_watcher(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """S5.6 carry-fwd S8.2 wiring: AlertWatcher must be constructed in
    composition (after migrations have run, so the boot kill_switch
    seed row is snapshotted and never fires a phantom flip)."""
    import datetime as dt

    from app.composition import compose_agent
    from observability.alerts import AlertWatcher

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
    ):
        loop = compose_agent(
            cfg,
            secrets=secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    assert loop.components.alert_watcher is not None
    assert isinstance(loop.components.alert_watcher, AlertWatcher)
    # The watcher must have already snapshotted the seed row's set_at
    # so a `poll()` immediately after composition does NOT fire
    # `kill_switch_flip`.
    fired = loop.components.alert_watcher.poll(now=dt.datetime.now(dt.UTC))
    assert "kill_switch_flip" not in fired, (
        "boot seed kill_switch_state row should NOT fire a phantom "
        "flip alert; AlertWatcher must snapshot at construction"
    )


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: ADR-009 §3.1 — boot-time PDTState wiring.
# ---------------------------------------------------------------------------


def _patch_broker_account(equity: float = 22_300.0):
    """Helper: returns a context manager that replaces
    `AlpacaBroker.get_account` with a stub returning a deterministic
    AccountSnapshot. compose_agent calls get_account once at boot to
    seed PDTState; the SDK is patched out at construction so we'd hit
    a MagicMock chain otherwise."""
    from broker.protocol import AccountSnapshot

    snap = AccountSnapshot(
        equity=equity, cash=equity, buying_power=equity * 2,
        long_market_value=0.0, daypl=0.0,
        snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        last_equity=equity, daytrade_count=0,
    )
    return patch("broker.alpaca.AlpacaBroker.get_account", return_value=snap)


def test_compose_agent_constructs_pdt_state_from_boot_account(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """ADR-009 §3.1 / S-PDT-2 AC-6: compose_agent calls broker.get_account
    once at boot and constructs a PDTState. Active flag reflects
    `last_equity < 25000` AND `pdt_enabled`.
    """
    from app.composition import compose_agent
    from execution.pdt_budget import PDTState

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
        _patch_broker_account(equity=22_300.0),
    ):
        loop = compose_agent(
            cfg,
            secrets=secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    pdt_state = loop.components.pdt_state
    assert isinstance(pdt_state, PDTState)
    assert pdt_state.is_active() is True


def test_compose_agent_pdt_state_inactive_above_threshold(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """`last_equity >= 25000` → inactive even with `pdt_enabled=True` (default)."""
    from app.composition import compose_agent

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
        _patch_broker_account(equity=27_500.0),
    ):
        loop = compose_agent(
            cfg, secrets=secrets, journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    assert loop.components.pdt_state is not None
    assert loop.components.pdt_state.is_active() is False


def test_compose_agent_reconciler_holds_pdt_state(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """The reconciler must be constructed with the same `PDTState`
    instance held on AgentComponents — otherwise the tick refresh
    target diverges from the consumer-visible state."""
    from app.composition import compose_agent

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
        _patch_broker_account(equity=22_300.0),
    ):
        loop = compose_agent(
            cfg, secrets=secrets, journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    rec = loop.components.reconciler
    assert rec._pdt_state is loop.components.pdt_state
    assert rec._pdt_enabled is True


def test_compose_agent_paper_vs_live_only_differs_on_credentials(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """AC-2 / D-007 sacred: switching broker_mode must change ONLY the
    broker's credential payload + endpoint, not the wired class identity
    or any other component shape."""
    from app.composition import compose_agent

    paper_secrets = _secrets(paper=True)
    live_secrets = _secrets(paper=False)
    journal = JournalRepo(db_path)

    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
    ):
        paper_loop = compose_agent(
            _config(broker_mode="paper", tmp_path=tmp_path),
            secrets=paper_secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim_paper"),
        )
        live_loop = compose_agent(
            _config(broker_mode="live", tmp_path=tmp_path),
            secrets=live_secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim_live"),
        )

    # Same broker class, different mode + endpoint.
    assert type(paper_loop.components.broker) is type(live_loop.components.broker)
    assert paper_loop.components.broker.mode == "paper"
    assert live_loop.components.broker.mode == "live"
    assert paper_loop.components.broker.endpoint == PAPER_ENDPOINT
    assert live_loop.components.broker.endpoint == LIVE_ENDPOINT

    # Every other component class is identical.
    for field in (
        "universe",
        "prefilter",
        "analyzer",
        "reviewer",
        "proposal_store",
        "execution",
        "reconciler",
        "killswitch",
    ):
        a = getattr(paper_loop.components, field)
        b = getattr(live_loop.components, field)
        assert type(a) is type(b), f"class differs for {field}: {type(a)} vs {type(b)}"


def test_compose_agent_wires_fill_ingestor_into_reconciler(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """WS1 (D-078, delta §1/§2.1): composition must wire a FillIngestor
    into the Reconciler ctor so fill outcomes are recorded at the top of
    every tick. Without this wiring the whole position-truth contract is
    dead code in prod."""
    from app.composition import compose_agent
    from reconciler.fill_ingest import FillIngestor

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
    ):
        loop = compose_agent(
            cfg,
            secrets=secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    rec = loop.components.reconciler
    ingestor = rec._fill_ingestor  # type: ignore[attr-defined]
    assert isinstance(ingestor, FillIngestor)
    # The ingestor polls the SAME broker and journal the reconciler uses.
    assert ingestor._broker is loop.components.broker  # type: ignore[attr-defined]
    assert ingestor._journal is journal  # type: ignore[attr-defined]


def test_compose_agent_wires_halt_repage_minutes_into_killswitch(
    db_path: str, prompts_dir: Path, tmp_path: Path
) -> None:
    """WS1 (D-078, delta §2.3): `killswitch.halt_repage_minutes` from
    config must reach the KillSwitch ctor (not silently stay at the
    class default)."""
    from app.composition import compose_agent

    cfg = _config(broker_mode="paper", tmp_path=tmp_path)
    cfg.killswitch.halt_repage_minutes = 90  # non-default
    secrets = _secrets(paper=True)
    journal = JournalRepo(db_path)
    with (
        patch("broker.alpaca.TradingClient"),
        patch("broker.alpaca.StockHistoricalDataClient"),
    ):
        loop = compose_agent(
            cfg,
            secrets=secrets,
            journal=journal,
            prompts_dir=prompts_dir,
            similarity_artifact_dir=str(tmp_path / "sim"),
        )

    ks = loop.components.killswitch
    assert ks._repage_minutes == 90  # type: ignore[attr-defined]
