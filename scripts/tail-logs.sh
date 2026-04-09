#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY="$SCRIPT_DIR/../keys/id_ed25519"

ssh -i "$KEY" -o StrictHostKeyChecking=no root@209.97.159.53 \
    "tail -f /opt/agent-scraper/logs/requests-\$(date -u +%Y-%m-%d).jsonl"
