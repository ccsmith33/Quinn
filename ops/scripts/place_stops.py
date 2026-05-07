"""Place protective sell-stop GTC orders for currently-held broker positions
using Quinn's original analyzer-chosen stop levels from proposals.raw_response.

Usage:
    python3 /tmp/place_stops.py            # dry-run, prints plan, no submission
    python3 /tmp/place_stops.py --apply    # actually submits the orders

Resolution strategy per held symbol:
    1. Find the latest `proposals` row where symbol matches AND raw_response
       JSON has a non-null `stop_loss_price`.
    2. If multiple, pick the most recent by created_at.
    3. If none found, the symbol is reported as UNRESOLVED (no order placed).
    4. Validate: stop must be strictly below current bid; flag any stop that
       would trigger immediately.

Outputs a per-symbol plan: held_qty | entry_avg | proposed_stop | stop_pct |
current_bid | status. Status is OK / NO_PROPOSAL / STOP_ABOVE_BID / etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.request
import urllib.error

JOURNAL_DB = "/var/lib/quinn/journal.db"


def alpaca_request(path: str, method: str = "GET", body: dict | None = None) -> dict | list:
    base = os.environ["ALPACA_ENDPOINT"].rstrip("/")
    headers = {
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, headers=headers, method=method, data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {body_text}") from e


def get_broker_positions() -> list[dict]:
    return alpaca_request("/v2/positions")  # type: ignore[return-value]


def get_quote_bid(symbol: str) -> float | None:
    base = os.environ["ALPACA_ENDPOINT"].rstrip("/")
    if "paper" in base:
        data_base = "https://data.alpaca.markets"
    else:
        data_base = "https://data.alpaca.markets"
    headers = {
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
    }
    req = urllib.request.Request(
        f"{data_base}/v2/stocks/{symbol}/quotes/latest",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        bid = data.get("quote", {}).get("bp")
        return float(bid) if bid else None
    except Exception:
        return None


def latest_stop_loss_for_symbol(conn: sqlite3.Connection, symbol: str) -> tuple[float, int, str] | None:
    """Returns (stop_loss_price, proposal_id, created_at) for the most recent
    proposal on this symbol with a stop_loss_price in raw_response."""
    rows = conn.execute(
        "SELECT id, raw_response, created_at FROM proposals "
        "WHERE symbol = ? ORDER BY created_at DESC",
        (symbol,),
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        # raw_response may be the parsed proposal dict directly, or wrap a
        # `trade` / `proposal` block. Try several shapes.
        for candidate in (payload, payload.get("trade"), payload.get("proposal")):
            if not isinstance(candidate, dict):
                continue
            sl = candidate.get("stop_loss_price")
            if sl is not None:
                try:
                    return float(sl), int(row[0]), str(row[2])
                except (TypeError, ValueError):
                    continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually submit orders")
    args = parser.parse_args()

    print(f"=== {'APPLY MODE — orders WILL be submitted' if args.apply else 'DRY-RUN — no orders submitted'} ===\n")

    positions = get_broker_positions()
    print(f"Broker positions: {len(positions)}\n")

    conn = sqlite3.connect(JOURNAL_DB)
    conn.row_factory = sqlite3.Row

    plan = []
    for pos in sorted(positions, key=lambda p: p["symbol"]):
        symbol = pos["symbol"]
        held_qty = int(pos["qty"])
        avg_entry = float(pos["avg_entry_price"])

        sl_info = latest_stop_loss_for_symbol(conn, symbol)
        bid = get_quote_bid(symbol)

        if sl_info is None:
            plan.append({
                "symbol": symbol,
                "qty": held_qty,
                "entry": avg_entry,
                "stop": None,
                "bid": bid,
                "proposal_id": None,
                "status": "NO_PROPOSAL",
            })
            continue

        stop_price, proposal_id, _created = sl_info
        # Round to 2dp; Alpaca rejects sub-cent stops on most equities.
        stop_price = round(stop_price, 2)

        status = "OK"
        if bid is not None and stop_price >= bid:
            status = f"STOP_ABOVE_BID(bid={bid:.2f})"
        elif stop_price >= avg_entry:
            status = f"STOP_ABOVE_ENTRY"

        plan.append({
            "symbol": symbol,
            "qty": held_qty,
            "entry": avg_entry,
            "stop": stop_price,
            "bid": bid,
            "proposal_id": proposal_id,
            "status": status,
        })

    # Print plan table
    print(f"{'SYMBOL':<6} {'QTY':>5} {'ENTRY':>8} {'STOP':>8} {'STOP%':>8} {'BID':>8} {'PROP':>6} STATUS")
    print("-" * 80)
    for p in plan:
        stop_str = f"{p['stop']:.2f}" if p["stop"] is not None else "-"
        bid_str = f"{p['bid']:.2f}" if p["bid"] is not None else "-"
        if p["stop"] is not None and p["entry"]:
            pct = (p["stop"] / p["entry"] - 1) * 100
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = "-"
        prop_str = str(p["proposal_id"]) if p["proposal_id"] else "-"
        print(f"{p['symbol']:<6} {p['qty']:>5} {p['entry']:>8.2f} {stop_str:>8} {pct_str:>8} {bid_str:>8} {prop_str:>6} {p['status']}")

    submittable = [p for p in plan if p["status"] == "OK"]
    issues = [p for p in plan if p["status"] != "OK"]
    print()
    print(f"Submittable: {len(submittable)}")
    print(f"Needs attention: {len(issues)}")
    for p in issues:
        print(f"  - {p['symbol']}: {p['status']}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to submit.")
        return 0

    # Live submission
    print("\n=== SUBMITTING ===")
    submitted = 0
    failed = 0
    for p in submittable:
        body = {
            "symbol": p["symbol"],
            "qty": str(p["qty"]),
            "side": "sell",
            "type": "stop",
            "time_in_force": "gtc",
            "stop_price": str(p["stop"]),
            "client_order_id": f"manual-stop-{p['symbol']}-{p['proposal_id']}",
        }
        try:
            resp = alpaca_request("/v2/orders", method="POST", body=body)
            print(f"  OK   {p['symbol']:<6} qty={p['qty']:>4} stop={p['stop']:.2f}  id={resp.get('id', '?')[:8]}")
            submitted += 1
        except Exception as e:
            print(f"  FAIL {p['symbol']:<6}: {e}")
            failed += 1

    print(f"\nSubmitted: {submitted}  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
