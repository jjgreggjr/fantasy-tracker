# Runs the weekly pull. Called by the scheduled task, or run it by hand.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$py = if ($env:FF_PYTHON) { $env:FF_PYTHON } else { "python" }
& $py -m ff.run_weekly @args
exit $LASTEXITCODE
