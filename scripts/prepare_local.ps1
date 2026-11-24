[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache' }
    uv sync --locked
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }

    $python = Join-Path $projectRoot '.venv/Scripts/python.exe'
    & $python scripts/download_weights.py
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint download failed.' }
    & $python scripts/prepare_coco2017.py
    if ($LASTEXITCODE -ne 0) { throw 'COCO2017 preparation failed.' }

    $source = 'runs/yolov5s_baseline/yolov5-v7.0'
    if (-not (Test-Path -LiteralPath $source)) {
        git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git $source
        if ($LASTEXITCODE -ne 0) { throw 'YOLOv5 checkout failed.' }
    }

    & $python scripts/run_yolov5s_baseline.py
    if ($LASTEXITCODE -ne 0) { throw 'Local FP32 baseline failed.' }
} finally {
    Pop-Location
}
