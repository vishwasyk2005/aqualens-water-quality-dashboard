$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$preparedRuntime = Join-Path $PSScriptRoot '../../work/runtime/packages'
$preparedProphet = Join-Path $PSScriptRoot '../../work/prophet_runtime'
$preparedTorch = Join-Path $env:USERPROFILE 'Documents/Codex/work/aqdeps'
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$bundledDataReady = $false
if (Test-Path -LiteralPath $bundledPython) {
    # Prefer the installed runtime's data libraries over development-only copies.
    $env:PYTHONPATH = ''
    if (Test-Path -LiteralPath (Join-Path $preparedTorch 'torch')) { $env:PYTHONPATH = $preparedTorch }
    & $bundledPython -c 'import pandas,numpy' 2>$null
    $bundledDataReady = $LASTEXITCODE -eq 0
}
if ($bundledDataReady) {
    $runPython = $bundledPython
} elseif ((Test-Path -LiteralPath $bundledPython) -and (Test-Path -LiteralPath (Join-Path $preparedRuntime 'pandas'))) {
    $runPython = $bundledPython
    $env:PYTHONPATH = (Resolve-Path -LiteralPath $preparedRuntime).Path
    if (Test-Path -LiteralPath (Join-Path $preparedTorch 'torch')) { $env:PYTHONPATH = $preparedTorch + ';' + $env:PYTHONPATH }
} else {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (Test-Path -LiteralPath $bundledPython) { & $bundledPython -m venv .venv }
        else { python -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 and retry.' }
    }
    $runPython = $venvPython
    & $runPython -c 'import pandas,numpy,torch,prophet' 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $runPython -m pip install -r requirements-training.txt
        if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies. Check your connection.' }
    }
}
if (Test-Path -LiteralPath (Join-Path $preparedProphet 'prophet')) {
    $prophetPath = (Resolve-Path -LiteralPath $preparedProphet).Path
    $env:PYTHONPATH = (@($prophetPath, $env:PYTHONPATH) | Where-Object { $_ }) -join ';'
}
$runtimeTemp = Join-Path $PSScriptRoot '.runtime-temp'
New-Item -ItemType Directory -Force -Path $runtimeTemp | Out-Null
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
Write-Host 'AquaLens dashboard: http://localhost:8765'
Write-Host 'Keep this window open while using the dashboard. Press Ctrl+C to stop.'
& $runPython server.py --port 8765
