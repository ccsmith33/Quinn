"""Operator dashboard FastAPI app (S9.1, D-063).

Read-only HTTP UI. Five Jinja2-rendered routes plus `/healthz`. HTTP basic
auth (constant-time compare) on every UI route. No state-changing endpoints
— halt/resume stays on Telegram + HMAC webhook (ADR-004).

Architectural notes:
- Same crash-isolation pattern as ADR-004 — runs in its own systemd unit,
  owns its own port (`cfg.dashboard.port`, default 8444), no shared
  process state with the agent loop, the Telegram bot, or the HMAC
  webhook listener.
- Reads ONLY through `JournalRepo`. New helpers were added to
  `src/journal/repo.py` (`count_filings_since`, `count_proposals_since`,
  `count_executions_since`, `get_account_snapshots_since`,
  `list_recent_proposals`, `list_recent_executions_with_orders`,
  `llm_spend_breakdown_mtd`). See repo.py docstrings for each.
- Optional `BrokerAdapter` for `/positions` live-quote enrichment with a
  5-second in-process TTL cache. Without one, the page falls back to the
  most-recent journal `positions` rows.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from broker.protocol import BrokerAdapter
from config.calendar import ET
from journal.repo import JournalRepo
from killswitch.api import KillSwitch, KillSwitchUninitialized
from observability.log_port import get_logger

from .auth import make_basic_auth_dependency

log = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_VALID_DECISION_STATUSES = {"accepted", "rejected"}


class _QuoteCache:
    """5-second TTL cache for broker quotes (AC-6)."""

    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, float]] = {}  # symbol -> (cached_at, last_price)

    def get(self, symbol: str, broker: BrokerAdapter, now: float) -> float | None:
        cached = self._store.get(symbol)
        if cached is not None and (now - cached[0]) < self._ttl:
            return cached[1]
        try:
            q = broker.get_quote(symbol)
        except Exception:  # broker offline / symbol unknown — degrade quietly
            log.warning("dashboard quote fetch failed", extra={"symbol": symbol})
            return None
        self._store[symbol] = (now, float(q.last))
        return float(q.last)


def _today_et_start(now: _dt.datetime) -> _dt.datetime:
    """Start of the current ET calendar day, returned as a UTC-aware datetime.

    The trading-day boundary is midnight America/New_York, not UTC midnight.
    Mirrors `jobs.daily_report._session_bounds_utc`.
    """
    et_now = now.astimezone(ET) if now.tzinfo is not None else now.replace(tzinfo=ET)
    et_start = _dt.datetime.combine(et_now.date(), _dt.time(0, 0), tzinfo=ET)
    return et_start.astimezone(_dt.UTC)


def _sparkline_points(snaps: list[Any], *, width: int = 600, height: int = 80) -> str:
    """Build the SVG `points=` attribute from a list of AccountSnapshotRow.

    Returns an empty string for fewer than 2 points (the template suppresses
    the SVG entirely in that case).
    """
    if len(snaps) < 2:
        return ""
    equities = [float(s.equity) for s in snaps]
    lo = min(equities)
    hi = max(equities)
    span = hi - lo or 1.0
    n = len(equities)
    pts = []
    for i, e in enumerate(equities):
        x = (i / (n - 1)) * width
        # Invert Y so larger equity is higher on the page.
        y = height - ((e - lo) / span) * height
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _position_row(
    journal: JournalRepo,
    *,
    symbol: str,
    qty: int,
    avg_entry_price: float,
    current_quote: float | None,
    unrealized_pnl: float,
    market_value: float,
) -> dict[str, Any]:
    """Assemble a single /positions row: live price-derived movement/PnL%
    plus the entry thesis and current stop/take-profit levels joined from
    the executions/orders/proposals tables via `get_position_entry_context`.
    """
    cost_basis = abs(avg_entry_price * qty)
    movement_pct: float | None = None
    if current_quote is not None and avg_entry_price:
        movement_pct = (current_quote / avg_entry_price - 1.0) * 100.0
    unrealized_pct: float | None = None
    if cost_basis > 0:
        unrealized_pct = (unrealized_pnl / cost_basis) * 100.0
    ctx = journal.get_position_entry_context(symbol)
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry_price": avg_entry_price,
        "current_quote": current_quote,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pct": unrealized_pct,
        "movement_pct": movement_pct,
        "market_value": market_value,
        "thesis": ctx.get("thesis") if ctx else None,
        "conviction": ctx.get("conviction") if ctx else None,
        "direction": ctx.get("direction") if ctx else None,
        "decision_id": ctx.get("decision_id") if ctx else None,
        "stop_price": ctx.get("stop_price") if ctx else None,
        "take_profit_price": ctx.get("take_profit_price") if ctx else None,
    }


def build_app(
    *,
    journal: JournalRepo,
    dashboard_user: str,
    dashboard_password: str,
    broker: BrokerAdapter | None = None,
    auto_refresh_seconds: int = 30,
) -> FastAPI:
    """Build the dashboard FastAPI app.

    `dashboard_user` / `dashboard_password` are captured by closure inside
    the auth dependency and never logged or returned in any response.
    """
    auth = make_basic_auth_dependency(dashboard_user, dashboard_password)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    quote_cache = _QuoteCache(ttl_seconds=5.0)

    app = FastAPI(title="quinn-dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    ks = KillSwitch(journal)

    def _ctx(**extra: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "auto_refresh_seconds": auto_refresh_seconds,
        }
        base.update(extra)
        return base

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        # Unauthenticated readiness probe — mirrors webhook listener.
        return JSONResponse({"status": "ok"})

    # -----------------------------------------------------------------
    # /  — overview (AC-3)
    # -----------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def overview(
        request: Request, _auth: None = Depends(auth)
    ) -> HTMLResponse:
        now = _dt.datetime.now(_dt.UTC)
        today = _today_et_start(now)
        try:
            ks_row = ks.state()
            ks_state = ks_row.state
            ks_reason = ks_row.reason
            ks_set_at = ks_row.set_at.isoformat() if ks_row.set_at else "—"
        except KillSwitchUninitialized:
            ks_state = "uninitialized"
            ks_reason = "—"
            ks_set_at = "—"

        snap = journal.get_latest_account_snapshot()
        equity = float(snap.equity) if snap is not None else None
        daypl = float(snap.daypl) if snap is not None else None

        snaps_30d = journal.get_account_snapshots_since(now - _dt.timedelta(days=30))
        sparkline = _sparkline_points(snaps_30d)

        return templates.TemplateResponse(
            request,
            "overview.html",
            _ctx(
                page_title="overview",
                active_route="overview",
                ks_state=ks_state,
                ks_reason=ks_reason,
                ks_set_at=ks_set_at,
                equity=equity,
                daypl=daypl,
                filings_today=journal.count_filings_since(today),
                proposals_today=journal.count_proposals_since(today),
                trades_today=journal.count_executions_since(
                    today, decision="accepted"
                ),
                auto_halt_status="active" if ks_state == "halted" else "ok",
                sparkline_points=sparkline,
            ),
        )

    # -----------------------------------------------------------------
    # /proposals  (AC-4)
    # -----------------------------------------------------------------
    @app.get("/proposals", response_class=HTMLResponse)
    async def proposals(
        request: Request,
        _auth: None = Depends(auth),
        symbol: str | None = Query(default=None, max_length=16),
        status_filter: str | None = Query(default=None, alias="status", max_length=16),
        limit: int = Query(default=100),
    ) -> HTMLResponse:
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be between 1 and 1000",
            )
        if (
            status_filter is not None
            and status_filter != ""
            and status_filter not in _VALID_DECISION_STATUSES
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"status must be one of {sorted(_VALID_DECISION_STATUSES)}",
            )
        sym = symbol or None
        if sym is not None and not sym.replace(".", "").replace("-", "").isalnum():
            # Defensive — rejects path-traversal-flavoured noise.
            sym = None
        eff_status = status_filter or None
        rows = journal.list_recent_proposals(
            limit=limit, symbol=sym, decision_status=eff_status
        )
        enriched: list[dict[str, Any]] = []
        for p, filing_ticker in rows:
            review = journal.get_review_for_proposal(p.id) if p.id else None
            execrow = journal.get_execution_for_proposal(p.id) if p.id else None
            display_thesis = p.thesis
            if display_thesis is None and p.raw_response:
                # No-trade rows store NULL in p.thesis; the reason lives in
                # raw_response.thesis_or_reason. Parse once per row.
                try:
                    parsed = _json.loads(p.raw_response)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    val = parsed.get("thesis_or_reason") or parsed.get("thesis")
                    if isinstance(val, str):
                        display_thesis = val
            enriched.append(
                {
                    "id": p.id,
                    "created_at": p.created_at.isoformat() if p.created_at else "",
                    "decision_id": p.decision_id,
                    "kind": p.kind,
                    "symbol": p.symbol,
                    "filing_ticker": filing_ticker,
                    "direction": p.direction,
                    "conviction": p.conviction,
                    "thesis": p.thesis,
                    "display_thesis": display_thesis,
                    "raw_response": p.raw_response,
                    "review": review,
                    "execution_decision": execrow.decision if execrow else None,
                }
            )
        return templates.TemplateResponse(
            request,
            "proposals.html",
            _ctx(
                page_title="proposals",
                active_route="proposals",
                rows=enriched,
                symbol=sym,
                status_filter=eff_status,
                limit=limit,
            ),
        )

    # -----------------------------------------------------------------
    # /trades  (AC-5)
    # -----------------------------------------------------------------
    @app.get("/trades", response_class=HTMLResponse)
    async def trades(
        request: Request, _auth: None = Depends(auth)
    ) -> HTMLResponse:
        triples = journal.list_recent_executions_with_orders(limit=100)
        rows: list[dict[str, Any]] = []
        for execrow, orders, prow in triples:
            for o in orders:
                slippage: float | None = None
                if (
                    o.realized_fill_price is not None
                    and o.pre_submission_bid is not None
                    and o.pre_submission_ask is not None
                ):
                    mid = (o.pre_submission_bid + o.pre_submission_ask) / 2.0
                    slippage = float(o.realized_fill_price) - mid
                rows.append(
                    {
                        "decided_at": (
                            execrow.decided_at.isoformat() if execrow.decided_at else ""
                        ),
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": o.qty,
                        "fill_price": (
                            float(o.realized_fill_price)
                            if o.realized_fill_price is not None
                            else None
                        ),
                        "slippage": slippage,
                        "decision": execrow.decision,
                        "reject_reason": execrow.reject_reason,
                        "proposal_decision_id": (
                            prow.decision_id if prow is not None else None
                        ),
                    }
                )
        return templates.TemplateResponse(
            request,
            "trades.html",
            _ctx(
                page_title="trades",
                active_route="trades",
                rows=rows,
            ),
        )

    # -----------------------------------------------------------------
    # /positions  (AC-6)
    # -----------------------------------------------------------------
    @app.get("/positions", response_class=HTMLResponse)
    async def positions(
        request: Request, _auth: None = Depends(auth)
    ) -> HTMLResponse:
        snap = journal.get_latest_account_snapshot()
        equity = float(snap.equity) if snap is not None else None
        rows: list[dict[str, Any]] = []
        total_market_value = 0.0
        if broker is not None:
            try:
                broker_positions = broker.get_positions()
            except Exception:
                log.warning("dashboard broker positions fetch failed")
                broker_positions = []
            now_ts = _dt.datetime.now(_dt.UTC).timestamp()
            for bp in broker_positions:
                cur = quote_cache.get(bp.symbol, broker, now_ts)
                rows.append(
                    _position_row(
                        journal,
                        symbol=bp.symbol,
                        qty=int(bp.qty),
                        avg_entry_price=float(bp.avg_entry_price),
                        current_quote=cur,
                        unrealized_pnl=float(bp.unrealized_pnl),
                        market_value=float(bp.market_value),
                    )
                )
                total_market_value += float(bp.market_value)
        else:
            for pr in journal.get_open_position_rows():
                rows.append(
                    _position_row(
                        journal,
                        symbol=pr.symbol,
                        qty=int(pr.qty),
                        avg_entry_price=float(pr.avg_entry_price),
                        current_quote=None,
                        unrealized_pnl=float(pr.unrealized_pnl),
                        market_value=float(pr.market_value),
                    )
                )
                total_market_value += float(pr.market_value)
        exposure_pct: float | None = None
        if equity and equity > 0:
            exposure_pct = (total_market_value / equity) * 100.0
        return templates.TemplateResponse(
            request,
            "positions.html",
            _ctx(
                page_title="positions",
                active_route="positions",
                rows=rows,
                equity=equity,
                exposure_pct=exposure_pct,
            ),
        )

    # -----------------------------------------------------------------
    # /llm  (AC-7)
    # -----------------------------------------------------------------
    @app.get("/llm", response_class=HTMLResponse)
    async def llm_view(
        request: Request, _auth: None = Depends(auth)
    ) -> HTMLResponse:
        now = _dt.datetime.now(_dt.UTC)
        breakdown = journal.llm_spend_breakdown_mtd(now)
        total = sum(float(r["cost_usd"]) for r in breakdown)
        return templates.TemplateResponse(
            request,
            "llm.html",
            _ctx(
                page_title="llm",
                active_route="llm",
                rows=breakdown,
                mtd_total_usd=total,
            ),
        )

    return app
