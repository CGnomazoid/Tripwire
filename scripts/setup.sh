#!/usr/bin/env bash
# One-command environment setup.
#
# Known gotcha (as of Sep 2026, macOS + iCloud Drive + uv): recent CPython
# point releases (3.13.12, 3.14.x) patched site.py to silently skip any
# .pth file with the macOS "hidden" (UF_HIDDEN) flag set. uv sets that flag
# on the .pth file it writes for this project's editable install, and it
# seems to reapply it unpredictably (possibly interacting with this repo
# living under ~/Library/Mobile Documents/com~apple~CloudDocs, i.e. iCloud
# Drive) - not just after `uv sync`/`uv add`, but sometimes on a plain
# `uv run` too. Net effect: `import supervisor` can randomly break with no
# error message at all.
#
# The durable fix (not just a one-time flag clear): use `./run` instead of
# `uv run` directly. `./run` passes --env-file .env, which sets PYTHONPATH
# to src/ - the interpreter applies PYTHONPATH before site.py's .pth
# scanning ever runs, so it's immune to whatever is flipping that flag.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
for pth in .venv/lib/python*/site-packages/supervisor.pth; do
  if [ -e "$pth" ]; then
    chflags nohidden "$pth" || true
  fi
done

echo "done. verifying import via ./run (the reliable path)..."
./run python -c "import supervisor; print('OK:', supervisor.__file__)"
