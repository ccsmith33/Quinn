"""HTTP listener process entry point (S7.3, ADR-004 webhook).

Wires the HMAC-signed webhook FastAPI app to uvicorn. Runs as its own
systemd unit (`quinn-http.service`) so a crash here does not affect the
agent loop or the Telegram bot — and vice versa (ADR-004 §webhook).

NFR-15: HMAC key is read once via `load_secrets()` and passed in by closure;
never logged.
NFR-17: binds to a single fixed port (`config.killswitch.webhook_port`).
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from config.loader import load_config
from config.secrets import load_secrets
from journal.repo import JournalRepo
from killswitch.api import KillSwitch
from killswitch.webhook import build_app

DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"


def main(db_path: str = DEFAULT_DB_PATH) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    secrets = load_secrets()
    journal = JournalRepo(db_path)
    ks = KillSwitch(journal)
    app = build_app(
        ks=ks,
        journal=journal,
        hmac_key=secrets.kill_switch_hmac_key.get_secret_value(),
        counter_path=cfg.killswitch.webhook_counter_path,
    )
    uvicorn.run(app, host="0.0.0.0", port=cfg.killswitch.webhook_port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH)
