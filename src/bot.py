"""Telegram bot process entry point (S7.2, ADR-004).

Long-polls Telegram and dispatches commands to `BotCore`. Runs as its own
systemd unit (`quinn-bot.service`) so that an agent-loop crash does not
stop kill-switch confirmations and an outage in this process does not stop
the agent loop (decoupled crash domains, ADR-004).

NFR-15: secrets are read once via `load_secrets()` and passed in; never
logged.
NFR-17: only egress is `api.telegram.org` (long-poll). The systemd unit
enforces this via `IPAddressAllow`.
"""

from __future__ import annotations

import logging
import sys

from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from config.secrets import load_secrets
from journal.repo import JournalRepo
from killswitch.api import KillSwitch
from killswitch.telegram_bot import LONG_POLL_INTERVAL_S, BotCore

DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"


def build_application(bot_token: str, core: BotCore) -> Application:
    """Construct a PTB Application with a single text-message handler.

    All commands route through `core.handle_command` which keeps the
    Telegram-facing surface small (and replaceable in tests).
    """
    app = Application.builder().token(bot_token).build()

    async def on_message(update: Update, _ctx: object) -> None:
        msg = update.effective_message
        if msg is None or msg.text is None or msg.chat is None:
            return
        reply = core.handle_command(chat_id=msg.chat.id, text=msg.text)
        if reply is not None:
            await msg.reply_text(reply.text)

    app.add_handler(MessageHandler(filters.TEXT, on_message))
    return app


def main(db_path: str = DEFAULT_DB_PATH) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    secrets = load_secrets()
    journal = JournalRepo(db_path)
    ks = KillSwitch(journal)
    core = BotCore(
        ks=ks,
        journal=journal,
        allowed_chat_id=int(secrets.telegram_operator_chat_id.get_secret_value()),
    )
    app = build_application(secrets.telegram_bot_token.get_secret_value(), core)
    app.run_polling(poll_interval=LONG_POLL_INTERVAL_S, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH)
