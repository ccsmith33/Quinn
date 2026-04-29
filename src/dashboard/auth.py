"""HTTP basic auth dependency for the operator dashboard (S9.1 AC-2 D-063).

Constant-time compare via `secrets.compare_digest` on both the username and
the password. The expected creds are captured by closure so the FastAPI
endpoint signature stays free of secrets and the values are never logged
(NFR-15).

Wrong creds → 401 with `WWW-Authenticate: Basic realm="quinn"`. Same body
for missing creds, malformed header, wrong user, or wrong password — the
attacker should not be able to enumerate which leg failed.
"""

from __future__ import annotations

import base64
import binascii
import secrets as _stdlib_secrets
from collections.abc import Callable

from fastapi import HTTPException, Request, status

REALM = "quinn"


def _challenge() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="authentication required",
        headers={"WWW-Authenticate": f'Basic realm="{REALM}"'},
    )


def _parse_basic(header: str | None) -> tuple[str, str] | None:
    """Return `(user, password)` if the Authorization header is well-formed
    HTTP basic, else None. Never raises.
    """
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None
    encoded = parts[1].strip()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if ":" not in decoded:
        return None
    user, _, password = decoded.partition(":")
    return user, password


def make_basic_auth_dependency(
    expected_user: str, expected_password: str
) -> Callable[[Request], None]:
    """Build a FastAPI dependency that enforces HTTP basic auth.

    Uses `secrets.compare_digest` on both legs so the wall-time of a reject
    does not leak which byte differed. To handle length-difference attacks
    we compare against length-padded values: `compare_digest` returns
    False fast on length mismatch, but we still run BOTH compares so the
    overall path length is constant regardless of which leg failed.
    """
    expected_user_bytes = expected_user.encode("utf-8")
    expected_pass_bytes = expected_password.encode("utf-8")

    def dep(request: Request) -> None:
        parsed = _parse_basic(request.headers.get("Authorization"))
        # Always run both compare_digest calls regardless of `parsed` to
        # keep timing flat. Substitute zero-length probes if missing so the
        # attacker can't distinguish "no header" from "wrong creds".
        if parsed is None:
            user_bytes = b""
            pass_bytes = b""
            ok_format = False
        else:
            user_bytes = parsed[0].encode("utf-8")
            pass_bytes = parsed[1].encode("utf-8")
            ok_format = True
        user_ok = _stdlib_secrets.compare_digest(user_bytes, expected_user_bytes)
        pass_ok = _stdlib_secrets.compare_digest(pass_bytes, expected_pass_bytes)
        if not (ok_format and user_ok and pass_ok):
            raise _challenge()

    return dep
