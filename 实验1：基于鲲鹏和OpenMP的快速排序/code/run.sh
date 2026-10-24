#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" != "quicksort" || $# -ne 2 ]]; then
    echo "Usage: bash run.sh quicksort THREADS" >&2
    exit 2
fi
binary="$script_dir/../../build/lab1/linux-release/quicksort"
if [[ ! -x "$binary" ]]; then
    binary="$script_dir/../../build/lab1/make/quicksort"
fi
exec "$binary" "$2" 8000000

