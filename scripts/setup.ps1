# Windows equivalent of setup.sh. Picks the CUDA extra when an Nvidia GPU
# is detected (via nvidia-smi), otherwise falls back to CPU-only PyTorch.
# Override with $env:SUPERVISOR_EXTRA if you want something else.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$extra = $env:SUPERVISOR_EXTRA
if (-not $extra) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $extra = "cuda"
    } else {
        $extra = "torch"
    }
}

Write-Host "installing with --extra $extra"
uv sync --extra $extra

Write-Host "done. verifying import via .\run.ps1 ..."
& "$PSScriptRoot\..\run.ps1" python -c "import supervisor; print('OK:', supervisor.__file__)"
