"""Operator dashboard package (S9.1 D-063).

Read-only HTTP UI giving the operator a single password-protected page to
glance at what Quinn is doing. State-changing operations stay on Telegram +
HMAC webhook (ADR-004); this package never writes.

Run as its own systemd unit (`quinn-dashboard.service`); separate process
from the agent loop, the Telegram bot, and the HMAC webhook listener
(ADR-004 crash-isolation pattern).
"""

from .app import build_app

__all__ = ["build_app"]
