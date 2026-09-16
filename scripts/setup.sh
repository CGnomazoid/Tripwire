#!/usr/bin/env bash
# One-command environment setup.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick the right optional-dependency group for this machine: MLX on Apple
# Silicon, CUDA+bitsandbytes on a machine with an Nvidia GPU (Windows/Linux),
# plain CPU-only PyTorch otherwise. Override with SUPERVISOR_EXTRA=<name> if
# you want something different (e.g. `torch` to force CPU-only on Linux).
if [ -n "${SUPERVISOR_EXTRA:-}" ]; then
  extra="$SUPERVISOR_EXTRA"
elif [ "$(uname -s)" = "Darwin" ]; then
  extra="mlx"
elif command -v nvidia-smi >/dev/null 2>&1; then
  extra="cuda"
else
  extra="torch"
fi

echo "installing with --extra $extra"
uv sync --extra "$extra"

echo "done. verifying import..."
uv run python -c "import supervisor; print('OK:', supervisor.__file__)"
