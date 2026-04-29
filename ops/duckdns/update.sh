#!/usr/bin/env bash
# DuckDNS auto-updater (S9.1 AC-11, D-063).
#
# Cron entry (run as root):
#   */5 * * * * /opt/quinn/app/ops/duckdns/update.sh >>/var/log/duckdns.log 2>&1
#
# Reads `DUCKDNS_TOKEN` and `DUCKDNS_DOMAIN` from the secrets envelope at
# /etc/quinn/secrets.env. Both are optional in `Secrets` — skip this
# install on a LAN-only / development box.
#
# DuckDNS will detect the source IP automatically; pass `&ip=` only if
# the operator wants to pin to a specific IPv4. The 5-minute cadence is
# DuckDNS's recommended minimum.

set -eu

ENV_FILE="${QUINN_SECRETS_ENV:-/etc/quinn/secrets.env}"

if [ ! -r "${ENV_FILE}" ]; then
    echo "duckdns/update.sh: secrets envelope not readable: ${ENV_FILE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
. "${ENV_FILE}"

if [ -z "${DUCKDNS_TOKEN:-}" ] || [ -z "${DUCKDNS_DOMAIN:-}" ]; then
    echo "duckdns/update.sh: DUCKDNS_TOKEN or DUCKDNS_DOMAIN not set; skipping" >&2
    exit 0
fi

# DuckDNS expects the bare subdomain (no .duckdns.org suffix).
DOMAIN_BARE="${DUCKDNS_DOMAIN%.duckdns.org}"

RESPONSE=$(curl -sS --max-time 15 \
    "https://www.duckdns.org/update?domains=${DOMAIN_BARE}&token=${DUCKDNS_TOKEN}&ip=" \
    || true)

if [ "${RESPONSE}" = "OK" ]; then
    printf '%s duckdns: OK %s\n' "$(date -Iseconds)" "${DOMAIN_BARE}"
else
    printf '%s duckdns: FAIL response=%q\n' "$(date -Iseconds)" "${RESPONSE}" >&2
    exit 1
fi
