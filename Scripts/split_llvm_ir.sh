#!/usr/bin/env bash
# Task 3 owns resume checks and publication of complete output directories.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${REGRAPH_PYTHON:-python3}" "$SCRIPT_DIR/split_functions.py" "$@"
