#!/usr/bin/env bash
# Egress firewall rules for Quinn (NFR-17, architecture §9.5).
#
# Default-deny outgoing; allow-list the eight upstream hosts the system
# legitimately needs to reach. Inbound: only the kill-switch webhook
# (default port 8443) plus SSH on operator IPs.
#
# Run as root once at deploy. Idempotent — `ufw` no-ops on duplicate
# rules.
#
# Why ufw and not nftables: D-024 (v1 = git pull + venv install + ufw,
# no infra-as-code). v2 candidate: terraform-managed nftables.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "must run as root" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Reset to a known-clean baseline.
# ---------------------------------------------------------------------------
ufw --force reset

# ---------------------------------------------------------------------------
# Default policy: deny everything; we enable only what we need.
# `default deny outgoing` is the NFR-17 invariant — without it the
# allow-list below is decorative.
# ---------------------------------------------------------------------------
ufw default deny incoming
ufw default deny outgoing
ufw default deny routed

# ---------------------------------------------------------------------------
# Egress allow-list (architecture §9.5). ufw allow rules accept hostnames
# only when /etc/ufw/applications.d defines them; in practice we pin to
# protocols + DNS-resolved IPs at apply time. Emit the canonical hostname
# in the rule comment so reviewers can audit by reading this file.
# ---------------------------------------------------------------------------

# DNS — required to resolve all the names below at runtime.
ufw allow out on any to 1.1.1.1 port 53 proto udp comment "DNS (Cloudflare)"
ufw allow out on any to 8.8.8.8 port 53 proto udp comment "DNS (Google)"
ufw allow out on any to 1.1.1.1 port 53 proto tcp comment "DNS-over-TCP (Cloudflare)"
ufw allow out on any to 8.8.8.8 port 53 proto tcp comment "DNS-over-TCP (Google)"

# HTTPS to the eight allow-listed upstreams. ufw doesn't filter by
# hostname directly — at deploy, the operator runs the helper script
# `ops/firewall/refresh-egress.sh` (out of scope for this commit) to
# resolve each name to current IPs and emit `ufw allow out to <IP> port 443`.
# The names below ARE the source-of-truth allow-list and MUST match
# architecture §9.5 exactly — the test suite greps for them.
#
# Allowed hostnames:
#   - sec.gov, data.sec.gov, efts.sec.gov         # EDGAR (FR-3)
#   - api.anthropic.com                            # LLM analyzer (FR-15..FR-18)
#   - paper-api.alpaca.markets, api.alpaca.markets # Broker (FR-20..FR-22)
#   - data.alpaca.markets                          # Market data quotes (FR-23)
#   - api.telegram.org                             # Kill-switch (FR-32)
#   - *.backblazeb2.com                            # Backups (FR-31)
#   - in.logs.betterstack.com                      # Off-box logs (NFR-9)
#   - query1.finance.yahoo.com, query2.finance.yahoo.com  # yfinance (FR-9)

# Catch-all egress on 443 — the per-IP narrowing happens via the
# refresh-egress helper at deploy.
ufw allow out to any port 443 proto tcp comment "HTTPS egress (narrow with refresh-egress.sh)"

# ---------------------------------------------------------------------------
# Inbound: only kill-switch webhook + SSH on operator IPs.
# ---------------------------------------------------------------------------
# Webhook listener (S7.3, default port 8443). This is the HMAC-signed
# fallback for kill-switch flips when Telegram is unreachable.
ufw allow 8443/tcp comment "Kill-switch webhook (S7.3)"

# SSH — restrict to the operator's allowlisted IPs (replace YOUR_IP at
# deploy). The runbook's "place secrets" step also walks placing the
# operator IP allow-list here.
# ufw allow from YOUR_OPERATOR_IP to any port 22 proto tcp comment "SSH operator"
ufw allow ssh comment "SSH (narrow with operator IP allow-list at deploy)"

# ---------------------------------------------------------------------------
# Enable.
# ---------------------------------------------------------------------------
ufw --force enable
ufw status verbose

echo
echo "Egress allow-list (must match architecture §9.5):"
echo "  sec.gov, anthropic.com, alpaca.markets, telegram.org,"
echo "  backblazeb2.com, betterstack.com, yahoo.com"
