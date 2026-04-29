"""Operator dashboard process entry point (S9.1, D-063).

Wires the dashboard FastAPI app to uvicorn. Runs as its own systemd unit
(`quinn-dashboard.service`) so a crash here does not affect the agent
loop, the Telegram bot, or the HMAC webhook listener (ADR-004 §webhook
crash-isolation pattern).

NFR-15: dashboard creds (and broker creds) are read once via
`load_secrets()` and passed in by closure; never logged.
NFR-17: binds to a single port (`config.dashboard.port`, default 8444),
host defaults to 127.0.0.1 so the only public path is via Caddy.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from broker.alpaca import AlpacaBroker
from config.loader import load_config
from config.secrets import load_secrets
from journal.repo import JournalRepo

from .app import build_app

DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"


def main(db_path: str = DEFAULT_DB_PATH) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config()
    secrets = load_secrets()
    journal = JournalRepo(db_path)
    broker: AlpacaBroker | None
    try:
        broker = AlpacaBroker(
            api_key_id=secrets.alpaca_api_key_id.get_secret_value(),
            api_secret=secrets.alpaca_api_secret_key.get_secret_value(),
            endpoint=secrets.alpaca_endpoint.get_secret_value(),
        )
    except Exception:  # broker not strictly required for the dashboard to render
        broker = None
    app = build_app(
        journal=journal,
        dashboard_user=secrets.dashboard_user.get_secret_value(),
        dashboard_password=secrets.dashboard_password.get_secret_value(),
        broker=broker,
        auto_refresh_seconds=cfg.dashboard.auto_refresh_seconds,
    )
    uvicorn.run(
        app,
        host=cfg.dashboard.bind_host,
        port=cfg.dashboard.port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH)
