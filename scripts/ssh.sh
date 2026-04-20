#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY="$SCRIPT_DIR/../keys/id_ed25519"

if [ -f "$KEY" ]; then
    ssh -i "$KEY" -o StrictHostKeyChecking=no root@209.97.159.53 "$@"
else
    ssh -o StrictHostKeyChecking=no root@209.97.159.53 "$@"
fi
