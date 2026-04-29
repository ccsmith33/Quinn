"""S7.3 — HMAC-signed webhook fallback listener tests.

Architecture references: ADR-004 §webhook (signed-message auth, replay
protection via monotonic counter), FR-32 (60s halt budget), NFR-15 (HMAC
key never logged), NFR-17 (single fixed inbound port).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journal.migrate import apply_migrations
from journal.repo import JournalRepo
from killswitch.api import KillSwitch
from killswitch.webhook import build_app

HMAC_KEY = "test-shared-secret-do-not-use-in-prod"


def _sig(body: bytes) -> str:
    return hmac.new(HMAC_KEY.encode("utf-8"), body, hashlib.sha256).hexdigest()


@pytest.fixture
def env(tmp_path: Path) -> dict[str, object]:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    journal = JournalRepo(str(db_path))
    ks = KillSwitch(journal)
    counter_path = tmp_path / "state" / "webhook_counter"
    app = build_app(
        ks=ks,
        journal=journal,
        hmac_key=HMAC_KEY,
        counter_path=counter_path,
    )
    return {
        "app": app,
        "client": TestClient(app),
        "ks": ks,
        "journal": journal,
        "counter_path": counter_path,
    }


# ---------------------------------------------------------------------------
# AC-2 / AC-7: signature + counter validation
# ---------------------------------------------------------------------------


def test_valid_signature_halts(env: dict[str, object]) -> None:
    body = json.dumps({"counter": 1, "reason": "operator iOS shortcut"}).encode("utf-8")
    resp = env["client"].post(  # type: ignore[index]
        "/kill-switch/halt",
        content=body,
        headers={"X-Quinn-Sig": _sig(body), "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert env["ks"].is_halted() is True  # type: ignore[index]
    state = env["ks"].state()  # type: ignore[index]
    assert state.reason == "manual:webhook"
    assert state.set_by == "operator"


def test_invalid_signature_rejected_state_unchanged(env: dict[str, object]) -> None:
    body = json.dumps({"counter": 1, "reason": "spoofed"}).encode("utf-8")
    resp = env["client"].post(  # type: ignore[index]
        "/kill-switch/halt",
        content=body,
        headers={"X-Quinn-Sig": "0" * 64, "Content-Type": "application/json"},
    )
    assert resp.status_code == 401
    assert env["ks"].is_halted() is False  # type: ignore[index]


def test_missing_signature_rejected(env: dict[str, object]) -> None:
    body = json.dumps({"counter": 1, "reason": "no sig"}).encode("utf-8")
    resp = env["client"].post(  # type: ignore[index]
        "/kill-switch/halt",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 401
    assert env["ks"].is_halted() is False  # type: ignore[index]


def test_stale_counter_rejected(env: dict[str, object]) -> None:
    # First request advances counter to 5
    b1 = json.dumps({"counter": 5, "reason": "first"}).encode("utf-8")
    r1 = env["client"].post(  # type: ignore[index]
        "/kill-switch/halt",
        content=b1,
        headers={"X-Quinn-Sig": _sig(b1)},
    )
    assert r1.status_code == 200
    # Resume so we can detect that a second halt would change something
    env["ks"].resume(set_by="operator")  # type: ignore[index]
    # Replay with counter=5 → 409
    r2 = env["client"].post(  # type: ignore[index]
        "/kill-switch/halt",
        content=b1,
        headers={"X-Quinn-Sig": _sig(b1)},
    )
    assert r2.status_code == 409
    assert env["ks"].is_halted() is False  # type: ignore[index]
    # Older counter also rejected
    b_older = json.dumps({"counter": 3, "reason": "older"}).encode("utf-8")
    r3 = env["client"].post(  # type: ignore[index]
        "/kill-switch/halt",
        content=b_older,
        headers={"X-Quinn-Sig": _sig(b_older)},
    )
    assert r3.status_code == 409
    assert env["ks"].is_halted() is False  # type: ignore[index]


def test_counter_advances_on_accept(env: dict[str, object]) -> None:
    counter_path: Path = env["counter_path"]  # type: ignore[assignment]
    for c in (1, 2, 3):
        env["ks"].resume(set_by="operator")  # type: ignore[index]
        b = json.dumps({"counter": c, "reason": f"step {c}"}).encode("utf-8")
        resp = env["client"].post(  # type: ignore[index]
            "/kill-switch/halt", content=b, headers={"X-Quinn-Sig": _sig(b)}
        )
        assert resp.status_code == 200
    # persisted last-seen counter is 3
    assert counter_path.exists()
    assert counter_path.read_text(encoding="utf-8").strip() == "3"


# ---------------------------------------------------------------------------
# AC-4: /healthz
# ---------------------------------------------------------------------------


def test_healthz_returns_200(env: dict[str, object]) -> None:
    resp = env["client"].get("/healthz")  # type: ignore[index]
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# AC-5: /status (HMAC-protected mirror of Telegram /status)
# ---------------------------------------------------------------------------


def test_status_endpoint_requires_signature(env: dict[str, object]) -> None:
    resp = env["client"].get("/status")  # type: ignore[index]
    assert resp.status_code == 401


def test_status_endpoint_returns_expected_shape(env: dict[str, object]) -> None:
    body = b""  # empty body for GET; HMAC is over body anyway
    resp = env["client"].get(  # type: ignore[index]
        "/status", headers={"X-Quinn-Sig": _sig(body)}
    )
    assert resp.status_code == 200
    data = resp.json()
    expected_keys = {
        "kill_switch_state",
        "day_pl",
        "open_positions",
        "last_filing_fetched_at",
        "last_proposal_at",
        "mtd_inference_cost_usd",
    }
    assert expected_keys.issubset(data.keys())
    assert data["kill_switch_state"] == "active"


# ---------------------------------------------------------------------------
# AC-7: independence from Telegram bot (ADR-004)
# ---------------------------------------------------------------------------


def test_independent_of_telegram_bot() -> None:
    """Webhook module must not import bot/telegram libs (separate crash domain)."""
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "killswitch"
        / "webhook.py"
    ).read_text(encoding="utf-8")
    forbidden = ["telegram", "bot."]
    for token in forbidden:
        assert token not in src, f"webhook.py imports {token!r} (couples to telegram)"


# ---------------------------------------------------------------------------
# Counter persistence survives restart
# ---------------------------------------------------------------------------


def test_counter_persists_across_app_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    journal = JournalRepo(str(db_path))
    ks = KillSwitch(journal)
    counter_path = tmp_path / "state" / "webhook_counter"

    # First app instance accepts counter=10
    app1 = build_app(ks=ks, journal=journal, hmac_key=HMAC_KEY, counter_path=counter_path)
    client1 = TestClient(app1)
    body = json.dumps({"counter": 10, "reason": "boot test"}).encode("utf-8")
    r1 = client1.post("/kill-switch/halt", content=body, headers={"X-Quinn-Sig": _sig(body)})
    assert r1.status_code == 200

    # Second app instance (simulated restart) must reject counter=10 as stale
    ks.resume(set_by="operator")
    app2 = build_app(ks=ks, journal=journal, hmac_key=HMAC_KEY, counter_path=counter_path)
    client2 = TestClient(app2)
    r2 = client2.post(
        "/kill-switch/halt", content=body, headers={"X-Quinn-Sig": _sig(body)}
    )
    assert r2.status_code == 409
    assert ks.is_halted() is False
