#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY="$SCRIPT_DIR/../keys/id_ed25519"
REMOTE="root@209.97.159.53"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"

echo "==> Creating directories..."
ssh $SSH_OPTS "$REMOTE" "mkdir -p /opt/agent-scraper/{logs,content}"

echo "==> Generating self-signed TLS certificate..."
ssh $SSH_OPTS "$REMOTE" 'openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
    -keyout /opt/agent-scraper/tls.key \
    -out /opt/agent-scraper/tls.crt \
    -days 3650 -nodes \
    -subj "/CN=209.97.159.53"'

echo "==> Installing tcpdump..."
ssh $SSH_OPTS "$REMOTE" "apt-get update -qq && apt-get install -y -qq tcpdump"

echo "==> Server setup complete!"
