#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
KEY="$SCRIPT_DIR/../keys/id_ed25519"
REMOTE="root@209.97.159.53"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"

echo "==> Building for linux/amd64..."
cd "$PROJECT_DIR"
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o agent-scraper ./cmd/agent-scraper

echo "==> Uploading binary..."
scp $SSH_OPTS agent-scraper "$REMOTE:/opt/agent-scraper/agent-scraper"

echo "==> Uploading service file..."
scp $SSH_OPTS configs/agent-scraper.service "$REMOTE:/etc/systemd/system/agent-scraper.service"

echo "==> Uploading logrotate config..."
scp $SSH_OPTS configs/logrotate.conf "$REMOTE:/etc/logrotate.d/agent-scraper"

echo "==> Restarting service..."
ssh $SSH_OPTS "$REMOTE" "systemctl daemon-reload && systemctl enable agent-scraper && systemctl restart agent-scraper"

echo "==> Status:"
ssh $SSH_OPTS "$REMOTE" "systemctl status agent-scraper --no-pager"

echo "==> Done!"
