#!/usr/bin/env bash
# One-command environment setup.
#
# Known gotcha (as of Sep 2026): recent CPython point releases (3.13.12,
# 3.14.x) patched site.py to silently skip any .pth file with the macOS
# "hidden" (UF_HIDDEN) flag set. uv sets that flag on the .pth file it
# writes for this project's editable install, on every `uv sync`/`uv add`.
# Net effect: `import supervisor` randomly breaks after any dependency
# change, with no error - site.py just silently drops the src/ path.
# Fix: clear the hidden flag after every sync. This script does that.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
for pth in .venv/lib/python*/site-packages/supervisor.pth; do
  if [ -e "$pth" ]; then
    chflags nohidden "$pth"
    echo "cleared hidden flag on $pth"
  fi
done
echo "done. verifying import..."
uv run python -c "import supervisor; print('OK:', supervisor.__file__)"
