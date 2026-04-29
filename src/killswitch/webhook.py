"""HMAC-signed webhook fallback for the kill-switch (S7.3, ADR-004).

Independent of the Telegram bot (separate process, separate crash domain).
The signed-message posture is what protects the endpoint, not network ACLs
(ADR-004 webhook section). Replay protection is a monotonic counter
persisted to disk; counters ≤ last-seen are rejected with 409.

Endpoints:
- `POST /kill-switch/halt`   — halts the system on valid sig + fresh counter
- `GET  /healthz`            — unauthenticated 200 for systemd readiness
- `GET  /status`             — HMAC-protected JSON status mirror (FR-34)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from journal.repo import JournalRepo
from observability.log_port import get_logger

from .api import KillSwitch, KillSwitchUninitialized

log = get_logger(__name__)

SIG_HEADER = "X-Quinn-Sig"


class CounterStore:
    """Filesystem-backed monotonic counter for webhook replay protection.

    Persisted to a single file (per ADR-004 / story dev-notes). The file is
    written atomically (write-tmp + rename) so a crash mid-write cannot
    corrupt the persisted value.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def last_seen(self) -> int:
        if not self._path.exists():
            return 0
        text = self._path.read_text(encoding="utf-8").strip()
        return int(text) if text else 0

    def advance(self, new_value: int) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(str(int(new_value)), encoding="utf-8")
        tmp.replace(self._path)


def _verify_sig(body: bytes, provided: str | None, hmac_key: str) -> bool:
    if not provided:
        return False
    expected = hmac.new(hmac_key.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided)


def build_app(
    ks: KillSwitch,
    journal: JournalRepo,
    hmac_key: str,
    counter_path: Path | str,
) -> FastAPI:
    """Construct the FastAPI app. The HMAC key is captured by closure and
    never logged or returned in any response (NFR-15).
    """
    counter = CounterStore(Path(counter_path))
    app = FastAPI(title="quinn-killswitch-webhook", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/kill-switch/halt")
    async def halt(request: Request) -> JSONResponse:
        body = await request.body()
        if not _verify_sig(body, request.headers.get(SIG_HEADER), hmac_key):
            log.warning(
                "webhook signature verification failed",
                extra={"event": "webhook.bad_sig"},
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad signature")
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="invalid json"
            ) from e
        try:
            counter_val = int(payload["counter"])
            reason = str(payload.get("reason", "")).strip() or "operator"
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing or invalid 'counter'",
            ) from e
        last = counter.last_seen()
        if counter_val <= last:
            log.warning(
                "webhook stale counter",
                extra={"event": "webhook.stale_counter", "got": counter_val, "last": last},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"stale counter (last={last})",
            )
        ks.halt(
            reason="manual:webhook",
            set_by="operator",
            notes=f"webhook reason={reason!r} counter={counter_val}",
        )
        counter.advance(counter_val)
        return JSONResponse(
            content={
                "status": "halted",
                "set_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "counter": counter_val,
            }
        )

    @app.get("/status")
    async def status_endpoint(request: Request) -> JSONResponse:
        body = await request.body()
        if not _verify_sig(body, request.headers.get(SIG_HEADER), hmac_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad signature")
        try:
            ks_state = ks.state().state
        except KillSwitchUninitialized:
            ks_state = "uninitialized"
        snap = journal.get_latest_account_snapshot()
        positions = journal.get_open_positions()
        last_filing = journal.get_last_filing_fetched_at()
        last_proposal = journal.get_last_proposal_created_at()
        now = _dt.datetime.now(_dt.UTC)
        mtd = journal.get_mtd_inference_cost_usd(now)
        return JSONResponse(
            content={
                "kill_switch_state": ks_state,
                "day_pl": snap.daypl if snap is not None else None,
                "equity": snap.equity if snap is not None else None,
                "open_positions": [
                    {"symbol": p.symbol, "qty": p.qty} for p in positions
                ],
                "last_filing_fetched_at": (
                    last_filing.isoformat() if last_filing is not None else None
                ),
                "last_proposal_at": (
                    last_proposal.isoformat() if last_proposal is not None else None
                ),
                "mtd_inference_cost_usd": mtd,
            }
        )

    return app
