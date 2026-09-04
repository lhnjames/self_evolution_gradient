#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$repo_dir/model_cache"
export TMPDIR="$repo_dir/.tmp"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_dir"
exec "$repo_dir/.venv/bin/python" -m self_evolve.runner --config config/smoke.yaml "$@"

