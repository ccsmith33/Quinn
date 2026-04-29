#!/usr/bin/env bash
# Local-only setup helper. Idempotent — safe to run multiple times.
#
# Creates the per-user data directories and appends local-friendly config
# overrides to config/quinn.toml so the operator's $HOME paths are used
# instead of the production /var/lib/quinn/* paths.
#
# Run from project root:
#   bash ops/scripts/local_setup.sh

set -euo pipefail

LOCAL_DATA="$HOME/quinn-data"
TOML="config/quinn.toml"

echo "=== Creating local data directories ==="
mkdir -p "$LOCAL_DATA/state" "$LOCAL_DATA/similarity"
echo "  $LOCAL_DATA/state"
echo "  $LOCAL_DATA/similarity"

echo
echo "=== Patching $TOML with local-friendly overrides ==="

# webhook_counter_path: replace any existing line under [killswitch] OR append.
if grep -q "^webhook_counter_path" "$TOML"; then
    sed -i "s|^webhook_counter_path.*|webhook_counter_path = \"$LOCAL_DATA/state/webhook_counter\"|" "$TOML"
    echo "  webhook_counter_path → $LOCAL_DATA/state/webhook_counter (replaced existing)"
else
    if grep -q "^\[killswitch\]" "$TOML"; then
        # [killswitch] section exists — insert the line right after the header.
        sed -i "/^\[killswitch\]/a webhook_counter_path = \"$LOCAL_DATA/state/webhook_counter\"" "$TOML"
        echo "  webhook_counter_path → $LOCAL_DATA/state/webhook_counter (added under existing [killswitch])"
    else
        # No [killswitch] section yet — append both.
        printf "\n[killswitch]\nwebhook_counter_path = \"%s/state/webhook_counter\"\n" "$LOCAL_DATA" >> "$TOML"
        echo "  webhook_counter_path → $LOCAL_DATA/state/webhook_counter (new [killswitch] section)"
    fi
fi

echo
echo "=== Verification ==="
echo "Config:"
grep -A1 '^\[killswitch\]' "$TOML" | sed 's/^/  /'
echo
echo "Directories:"
ls -ld "$LOCAL_DATA/state" "$LOCAL_DATA/similarity" | sed 's/^/  /'

echo
echo "Done. Now retry the HTTP listener:"
echo "  bash -c 'set -a; source .env; set +a; exec .venv/bin/python -m src.http_listener'"
