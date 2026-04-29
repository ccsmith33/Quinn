# Quinn — Rehydration runbook

**Target**: NFR-7 — full system rehydrated from git + B2 backup in **≤ 60 min**.

This runbook is the proof that Quinn can be rebuilt from version control plus the latest B2 snapshot. Every step here is checked into the repo or scripted; the only inputs the operator provides at runtime are the fresh droplet credentials and the secrets envelope (which lives outside the repo by design — NFR-15).

D-019 (a)(b)(d) — systemd auto-restart, off-box logging deployment, daily backups — become VERIFIED on first successful execution of this runbook.

---

## 0. Prerequisites

- A DigitalOcean account (or any cloud provider; Quinn is OS-vanilla — Ubuntu 22.04 LTS or 24.04 LTS).
- The latest `secrets.env` envelope (operator-private; out-of-band — sealed inside 1Password / KeePassXC).
- The Backblaze B2 application key with read+download on the backup bucket.
- This repo cloned locally so the operator can copy `ops/` files onto the droplet.

---

## 1. Provision the droplet (target: 5 min)

```bash
# DO control panel: 1 vCPU / 1 GB / Ubuntu 24.04 LTS / Frankfurt or NYC1
# Note the public IP. Add the operator's SSH public key during creation.
ssh root@<droplet-ip>
```

If using Terraform instead of the control panel, see `ops/terraform/` (out of scope for v1; the control-panel path is what NFR-7 timing assumes).

---

## 2. Install OS dependencies (target: 5 min)

```bash
apt-get update
apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    sqlite3 \
    ufw \
    git \
    gzip
# Vector for off-box log shipping (S8.1).
curl -1sLf https://repositories.timber.io/public/vector/cfg/setup/bash.deb.sh | bash
apt-get install -y vector
```

---

## 3. Clone the repo + create the unprivileged user (target: 3 min)

```bash
useradd --system --create-home --shell /usr/sbin/nologin quinn
mkdir -p /opt/quinn
git clone https://github.com/ccsmith33/Quinn.git /opt/quinn/app
chown -R quinn:quinn /opt/quinn
mkdir -p /var/lib/quinn /var/log/quinn /etc/quinn
chown quinn:quinn /var/lib/quinn /var/log/quinn
```

---

## 4. Install Python deps (target: 3 min)

```bash
cd /opt/quinn/app
sudo -u quinn python3.11 -m venv .venv
sudo -u quinn make install
```

`make install` runs `pip install -e ".[dev]"` and applies migrations.

---

## 5. Place secrets in `/etc/quinn/secrets.env` (target: 2 min)

```bash
install -m 0600 -o root -g quinn /tmp/secrets.env /etc/quinn/secrets.env
# Verify NFR-15: file is mode 0600, root-owned, group quinn (read-only).
stat -c '%a %U %G' /etc/quinn/secrets.env  # expect: 600 root quinn
```

The envelope must contain ALL the keys listed in `src/config/secrets.py`. A missing key is a hard fail at process start (S1.4 `MissingSecret` exception).

---

## 6. Restore the latest journal from B2 (target: 5 min)

The B2 path layout is `b2://<bucket>/quinn/journal/YYYY/MM/DD-journal.db.gz`. Most-recent date wins.

```bash
sudo -u quinn make restore-from-b2
# Equivalent to:
#   .venv/bin/python -m src.jobs.restore_from_b2 --target /var/lib/quinn/journal.db
```

Verify the schema version recorded in the restored journal matches the migration files in this checkout:

```bash
sudo -u quinn make verify-schema
# Reads PRAGMA user_version + meta.schema_version; must equal the
# highest-numbered migration in src/journal/migrations/.
```

If `verify-schema` fails: the backup was taken at an older code revision than the deployed checkout. STOP — the operator decides whether to re-run pending migrations or to roll forward by re-deploying the matching commit.

---

## 7. Validate the Vector log-shipping config (S8.1 L-2 carry-forward)

```bash
sudo -u quinn cp /opt/quinn/app/ops/vector/quinn.toml /etc/vector/quinn.toml
vector validate /etc/vector/quinn.toml
# Must print "Validated".
```

This MUST pass before the next step — a syntax error here would otherwise only surface at first boot.

---

## 8. Orphan-sweep `<artifact_dir>` (S4.2 L-2 carry-forward)

The similarity-prefilter prune step (S4.2) commits the DELETE before unlinking the on-disk pickle artifacts. If the agent crashed mid-prune in the previous deployment, orphan files may exist under `/var/lib/quinn/similarity/` whose `(cik, form_type, accession_number)` triple is absent from the `similarity_cache` table.

```bash
sudo -u quinn .venv/bin/python -m src.prefilter.orphan_sweep \
    --root /var/lib/quinn/similarity \
    --dry-run     # remove --dry-run after the operator confirms the listing
```

The sweep is read-only by default; pass `--apply` to actually unlink. Sweep latency is bounded by the count of files under the root — typically sub-second on a fresh rehydrate.

---

## 9. Install systemd units + firewall + enable (target: 5 min)

```bash
install -m 0644 -o root -g root /opt/quinn/app/ops/systemd/*.service /etc/systemd/system/
install -m 0644 -o root -g root /opt/quinn/app/ops/systemd/*.timer /etc/systemd/system/

bash /opt/quinn/app/ops/firewall/ufw-rules.sh

systemctl daemon-reload
systemctl enable --now \
    quinn-agent.service \
    quinn-bot.service \
    quinn-http.service \
    quinn-dashboard.service \
    quinn-universe.timer \
    quinn-daily-report.timer \
    quinn-backup.timer

systemctl status quinn-*
# All services should be `active (running)`; all timers `active (waiting)`.
```

If running the dashboard on a public-internet droplet (S9.1 AC-10/AC-11):

```bash
# Source the secrets envelope so $DUCKDNS_DOMAIN is set in this shell
# before the Caddy substitution below. Skip if the operator already did
# `set -a; . /etc/quinn/secrets.env; set +a` for an earlier step.
set -a; . /etc/quinn/secrets.env; set +a

# Caddy: HTTPS reverse-proxy in front of the dashboard
apt-get install -y caddy
install -m 0644 -o root -g root /opt/quinn/app/ops/caddy/Caddyfile /etc/caddy/Caddyfile
sed -i "s/<your-name>.duckdns.org/${DUCKDNS_DOMAIN}/" /etc/caddy/Caddyfile
systemctl reload caddy

# DuckDNS: 5-minute cron auto-updater for the dynamic-DNS record
echo '*/5 * * * * root /opt/quinn/app/ops/duckdns/update.sh >>/var/log/duckdns.log 2>&1' \
    > /etc/cron.d/quinn-duckdns
chmod 0644 /etc/cron.d/quinn-duckdns
```

---

## 10. Smoke-test (target: 5 min)

- `systemctl status quinn-agent` → active
- Telegram: send `/status` to the bot, expect a reply within 10s
- Curl the webhook listener: `curl https://<droplet>:8443/status` should answer (HMAC-protected; just probing reachability)
- Curl the dashboard healthz: `curl http://127.0.0.1:8444/healthz` → 200 `{"status":"ok"}` (S9.1)
- Browser-load `https://${DUCKDNS_DOMAIN}/` (or `http://127.0.0.1:8444/` over an SSH tunnel for LAN-only): basic-auth challenge → enter `DASHBOARD_USER` + `DASHBOARD_PASSWORD` → overview page renders within 2s.
- Check Better Stack: a fresh log line tagged `service=quinn` should appear within 30s
- Trigger an explicit backup smoke: `sudo -u quinn make backup-now` — a new `quinn/journal/YYYY/MM/DD-journal.db.gz` should appear in B2 within 2 min, and a `backups` row should be inserted in `journal.db`.

---

## 11. Total time budget — NFR-7 60-minute target

| Step | Target | Notes |
|------|--------|-------|
| 1. Provision | 5 min | DO control panel image boot |
| 2. OS deps | 5 min | apt + Vector repo install |
| 3. Repo + user | 3 min | git clone over public network |
| 4. Python deps | 3 min | pip install -e .[dev] |
| 5. Secrets | 2 min | envelope copy |
| 6. Restore + verify | 5 min | B2 download + schema check |
| 7. Vector validate | 1 min | quick syntax check |
| 8. Orphan-sweep | 1 min | dry-run + apply |
| 9. systemctl | 5 min | unit install + enable |
| 10. Smoke | 5 min | end-to-end probes |
| **Total target** | **35 min** | safety margin against the 60 min NFR-7 ceiling |

The 60 min budget is the SLA; the procedural target is **≤ 40 min** so a single anomaly (slow B2 download, transient apt mirror) doesn't blow the SLA.

---

## 12. Rehydration log — recorded timed dry-run

Each operator-executed rehydration drill records its measured time below. Update on every drill; deletion-by-row only if the drill was abandoned mid-stream (note the reason).

| Date (UTC)  | Operator | Wall time | Notes |
|-------------|----------|-----------|-------|
| 2026-04-29  | dev-e8 (theoretical scripted dry-run) | TBD (operator-action) | Initial drill pending; runbook validated structurally by `tests/ops/test_systemd_units.py`. Operator must execute on a real droplet and update this row before S8.3 review approves. |

**Note for reviewer-e8 / operator**: AC-6 requires this table to have at least one real timed entry before S8.3 closes. The test suite asserts the table EXISTS and the runbook contains the timing target, but does not (and cannot) verify the wall-clock time was actually achieved. That's the manual exit criterion.

---

## 13. Failure modes during rehydration

- **B2 download empty** → check the operator's B2 application-key permissions. The bucket name is `$BACKUP_B2_BUCKET` from the secrets envelope.
- **`make verify-schema` fails** → the backup was taken at an older code revision. Re-deploy the matching tag (find via `git log src/journal/migrations/`).
- **Vector validate fails** → syntax error in `ops/vector/quinn.toml`; do NOT proceed (logs would silently drop).
- **Orphan-sweep finds orphans** → review the listing first via `--dry-run`; orphans are file-leak only, not data-correctness, so deferring sweep is safe but will grow disk usage.
- **Telegram /status no reply** → check `journalctl -u quinn-bot.service` for `auth` errors; rotate `TELEGRAM_BOT_TOKEN` if leaked.
