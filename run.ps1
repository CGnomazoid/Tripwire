# Windows equivalent of ./run - see that file for background. Passes
# --env-file .env so PYTHONPATH=src is set before anything else runs.
Set-Location $PSScriptRoot
uv run --env-file .env @args
