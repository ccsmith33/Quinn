"""S8.3 — systemd units + firewall + runbook tests (NFR-4, NFR-17, ACs 3-6).

TDD. Covers:
  AC-3: every unit has Restart=on-failure + RestartSec=10 +
        StartLimitIntervalSec=60 + StartLimitBurst=5 + hardening flags.
  AC-4: ufw rules match the architecture §9.5 egress allow-list.
  AC-5: rehydration runbook exists, is non-empty, and contains the
        required prerequisite steps.
  AC-6: timed dry-run is recorded somewhere the operator can review.

We don't shell out to `systemd-analyze` (not in dev path) — the tests
parse the unit files structurally, which is robust and CI-portable.
The architect's note in §9.3 makes the unit shape explicit, so a
parser-based check is sound.
"""

from __future__ import annotations

from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[2] / "ops"
SYSTEMD = OPS / "systemd"
FIREWALL = OPS / "firewall" / "ufw-rules.sh"
RUNBOOK = OPS / "runbooks" / "rehydrate.md"

REQUIRED_SERVICES = [
    "quinn-agent.service",
    "quinn-bot.service",
    "quinn-http.service",
    "quinn-universe.service",
    "quinn-daily-report.service",
    "quinn-backup.service",
    "quinn-dashboard.service",
]
REQUIRED_TIMERS = [
    "quinn-universe.timer",
    "quinn-daily-report.timer",
    "quinn-backup.timer",
]


def _parse_unit(text: str) -> dict[str, dict[str, str]]:
    """Very small INI-style parser. Returns {section: {key: value}}.

    Handles repeated keys by appending: subsequent values join with `\n`.
    """
    out: dict[str, dict[str, str]] = {}
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            out.setdefault(section, {})
            continue
        if section is None:
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k in out[section]:
            out[section][k] = out[section][k] + "\n" + v
        else:
            out[section][k] = v
    return out


# ---------------------------------------------------------------------------
# AC-3 — services
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_SERVICES)
def test_service_unit_exists(name: str) -> None:
    p = SYSTEMD / name
    assert p.is_file(), f"missing unit file: {p}"


@pytest.mark.parametrize("name", REQUIRED_SERVICES)
def test_service_has_required_sections(name: str) -> None:
    cfg = _parse_unit((SYSTEMD / name).read_text())
    assert "Unit" in cfg, f"{name}: missing [Unit]"
    assert "Service" in cfg, f"{name}: missing [Service]"


@pytest.mark.parametrize("name", REQUIRED_SERVICES)
def test_service_has_restart_policy(name: str) -> None:
    """NFR-4: Restart=on-failure + 10s wait + ≤ 5 restarts/min cap.

    `StartLimitIntervalSec` / `StartLimitBurst` belong in `[Unit]`, not
    `[Service]`: systemd silently ignores them in `[Service]` (they are
    Unit-scoped properties), which means the rate cap had no effect on
    the prior layout. The other two restart-policy keys are correctly
    `[Service]`-scoped per the systemd manual.
    """
    cfg = _parse_unit((SYSTEMD / name).read_text())
    svc = cfg["Service"]
    unit = cfg.get("Unit", {})
    assert svc.get("Restart") == "on-failure", svc.get("Restart")
    assert svc.get("RestartSec") == "10", svc.get("RestartSec")
    assert unit.get("StartLimitIntervalSec") == "60", (
        f"{name}: StartLimitIntervalSec must live in [Unit], got "
        f"{unit.get('StartLimitIntervalSec')!r}"
    )
    assert unit.get("StartLimitBurst") == "5", (
        f"{name}: StartLimitBurst must live in [Unit], got "
        f"{unit.get('StartLimitBurst')!r}"
    )


@pytest.mark.parametrize("name", REQUIRED_SERVICES)
def test_service_has_hardening_flags(name: str) -> None:
    """Architecture §9.3 hardening: NoNewPrivileges, ProtectSystem,
    ProtectHome, ReadWritePaths."""
    cfg = _parse_unit((SYSTEMD / name).read_text())
    svc = cfg["Service"]
    assert svc.get("NoNewPrivileges") == "true", svc.get("NoNewPrivileges")
    assert svc.get("ProtectSystem") == "strict", svc.get("ProtectSystem")
    assert svc.get("ProtectHome") == "true", svc.get("ProtectHome")
    rwpaths = svc.get("ReadWritePaths") or ""
    assert "/var/lib/quinn" in rwpaths, rwpaths
    assert "/var/log/quinn" in rwpaths, rwpaths


@pytest.mark.parametrize("name", REQUIRED_SERVICES)
def test_service_runs_as_quinn_user(name: str) -> None:
    cfg = _parse_unit((SYSTEMD / name).read_text())
    svc = cfg["Service"]
    assert svc.get("User") == "quinn", svc.get("User")
    assert svc.get("Group") == "quinn", svc.get("Group")


@pytest.mark.parametrize("name", REQUIRED_SERVICES)
def test_service_loads_secrets_env(name: str) -> None:
    cfg = _parse_unit((SYSTEMD / name).read_text())
    assert cfg["Service"].get("EnvironmentFile") == "/etc/quinn/secrets.env"


def test_agent_service_runs_src_app() -> None:
    """quinn-agent.service starts `python -m src.app` per architecture §9.3.
    S5.6 is currently in-flight (task #41); this test asserts the unit
    references the file even before S5.6 lands — landing S5.6 is what
    makes the smoke test runnable, not what makes this unit valid."""
    cfg = _parse_unit((SYSTEMD / "quinn-agent.service").read_text())
    exec_start = cfg["Service"].get("ExecStart", "")
    assert "src.app" in exec_start or "src/app.py" in exec_start, exec_start


def test_bot_service_runs_src_bot() -> None:
    cfg = _parse_unit((SYSTEMD / "quinn-bot.service").read_text())
    exec_start = cfg["Service"].get("ExecStart", "")
    assert "src.bot" in exec_start or "src/bot.py" in exec_start, exec_start


def test_http_service_runs_src_http_listener() -> None:
    cfg = _parse_unit((SYSTEMD / "quinn-http.service").read_text())
    exec_start = cfg["Service"].get("ExecStart", "")
    assert "http_listener" in exec_start, exec_start


def test_dashboard_service_runs_src_dashboard() -> None:
    """S9.1 AC-9 — operator dashboard unit starts `python -m src.dashboard`.
    Same hardening pattern as the other services (§9.3).
    """
    cfg = _parse_unit((SYSTEMD / "quinn-dashboard.service").read_text())
    exec_start = cfg["Service"].get("ExecStart", "")
    assert "src.dashboard" in exec_start, exec_start


# ---------------------------------------------------------------------------
# AC-3 — timers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_TIMERS)
def test_timer_unit_exists(name: str) -> None:
    p = SYSTEMD / name
    assert p.is_file(), f"missing timer file: {p}"


@pytest.mark.parametrize("name", REQUIRED_TIMERS)
def test_timer_has_oncalendar(name: str) -> None:
    cfg = _parse_unit((SYSTEMD / name).read_text())
    assert "Timer" in cfg, f"{name}: missing [Timer]"
    assert cfg["Timer"].get("OnCalendar"), cfg["Timer"]


def test_universe_timer_runs_at_07_eastern() -> None:
    cfg = _parse_unit((SYSTEMD / "quinn-universe.timer").read_text())
    on_cal = cfg["Timer"]["OnCalendar"]
    assert "07:00" in on_cal, on_cal


def test_daily_report_timer_runs_at_16_30_eastern() -> None:
    cfg = _parse_unit((SYSTEMD / "quinn-daily-report.timer").read_text())
    on_cal = cfg["Timer"]["OnCalendar"]
    assert "16:30" in on_cal, on_cal


def test_backup_timer_runs_at_02_eastern() -> None:
    cfg = _parse_unit((SYSTEMD / "quinn-backup.timer").read_text())
    on_cal = cfg["Timer"]["OnCalendar"]
    assert "02:00" in on_cal, on_cal


# ---------------------------------------------------------------------------
# AC-4 — firewall rules
# ---------------------------------------------------------------------------


def test_firewall_rules_file_exists() -> None:
    assert FIREWALL.is_file(), f"missing {FIREWALL}"


_EGRESS_HOSTS = [
    "sec.gov",
    "anthropic.com",
    "alpaca.markets",
    "telegram.org",
    "backblazeb2.com",
    "betterstack.com",
    "yahoo.com",
]


@pytest.mark.parametrize("host", _EGRESS_HOSTS)
def test_firewall_allows_egress_host(host: str) -> None:
    text = FIREWALL.read_text()
    assert host in text, f"egress allow-list missing {host}"


def test_firewall_default_egress_deny() -> None:
    """The architecture §9.5 egress invariant requires default-deny on
    egress; the script must set that before adding the allow rules."""
    text = FIREWALL.read_text()
    # ufw default deny outgoing, OR an explicit comment block setting it.
    assert "default deny outgoing" in text.lower() or "default-deny-outgoing" in text


def test_firewall_inbound_only_8443_and_ssh() -> None:
    text = FIREWALL.read_text()
    # Webhook port 8443 inbound
    assert "8443" in text
    # SSH inbound
    assert "ssh" in text.lower() or "22" in text


# ---------------------------------------------------------------------------
# AC-5 — rehydration runbook
# ---------------------------------------------------------------------------


def test_runbook_exists() -> None:
    assert RUNBOOK.is_file(), f"missing {RUNBOOK}"


def test_runbook_has_required_sections() -> None:
    """The runbook must walk an operator from a fresh droplet to a fully
    running Quinn deployment in ≤ 60 min (NFR-7). Specific steps required
    by AC-5 dev-note + carry-forwards:
      - Provision droplet
      - Install OS deps (incl. vector — S8.1 L-2)
      - Clone + make install
      - Place secrets in /etc/quinn/secrets.env
      - Restore latest backup from B2
      - Verify schema (`make verify-schema`)
      - vector validate (S8.1 L-2)
      - Orphan-sweep (S4.2 L-2)
      - systemctl enable --now quinn-*
      - 60-minute time target documented
    """
    text = RUNBOOK.read_text().lower()
    required = [
        "provision",
        "make install",
        "/etc/quinn/secrets.env",
        "restore",
        "make verify-schema",
        "vector validate",  # S8.1 L-2 carry-forward
        "orphan",  # S4.2 L-2 carry-forward
        "systemctl",
        "60 min",
    ]
    missing = [tok for tok in required if tok not in text]
    assert not missing, f"runbook missing required sections: {missing}"


def test_runbook_documents_b2_restore_path() -> None:
    """The restore step must reference the B2 path layout we use for
    backups: `quinn/journal/YYYY/MM/DD-journal.db.gz`."""
    text = RUNBOOK.read_text()
    assert "quinn/journal" in text, text[:500]


# ---------------------------------------------------------------------------
# AC-6 — timed rehydration record
# ---------------------------------------------------------------------------


def test_timed_dry_run_recorded() -> None:
    """AC-6: the operator timed-rehydrate exercise is recorded in the
    runbook (or in a sibling "rehydration-log" file). The record proves
    NFR-7 60-min target is achievable, not just documented."""
    text = RUNBOOK.read_text()
    # Look for either a "Recorded run" / "Timed run" heading OR an
    # explicit timing block. The runbook owner can update the timing on
    # each future drill; the assertion is that the section EXISTS.
    assert any(
        token.lower() in text.lower()
        for token in ["timed dry-run", "timed run", "recorded run", "## rehydration log"]
    ), text[:500]


# ---------------------------------------------------------------------------
# AC-3 (extension) — Makefile targets used by the runbook exist
# ---------------------------------------------------------------------------


def test_makefile_has_install_target() -> None:
    mk = (Path(__file__).resolve().parents[2] / "Makefile").read_text()
    assert "install:" in mk


def test_makefile_has_verify_schema_target() -> None:
    mk = (Path(__file__).resolve().parents[2] / "Makefile").read_text()
    assert "verify-schema:" in mk


def test_makefile_has_restore_from_b2_target() -> None:
    mk = (Path(__file__).resolve().parents[2] / "Makefile").read_text()
    assert "restore-from-b2:" in mk
