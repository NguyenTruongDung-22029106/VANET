#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8090}"
cd "$ROOT_DIR"

echo "Serving DASH demo root at http://127.0.0.1:${PORT}/player/index.html"
python3 -m http.server "$PORT"
