#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_root="${1:-/data/hanning/agent_self_evolution_gradient_bundle_20260901}"

if [[ -e "$bundle_root" ]]; then
  echo "Refusing to overwrite existing destination: $bundle_root" >&2
  exit 2
fi

mkdir -p "$bundle_root"
rsync -a \
  --exclude='/model_cache/' \
  --exclude='/.venv/' \
  --exclude='/.pip-cache/' \
  --exclude='/.pytest_cache/' \
  --exclude='/.tmp/' \
  --exclude='/.git/' \
  --exclude='/**/.git/' \
  --exclude='**/__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.safetensors' \
  --exclude='*.gguf' \
  --exclude='pytorch_model*.bin' \
  "$repo_root/" "$bundle_root/"

(
  cd "$bundle_root"
  find . -type f ! -name ARTIFACT_SHA256SUMS.txt -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > ARTIFACT_SHA256SUMS.txt
)

echo "Portable bundle created at $bundle_root"
