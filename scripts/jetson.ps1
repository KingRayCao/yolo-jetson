[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('sync', 'check', 'benchmark', 'collect')]
    [string]$Action,
    [switch]$CodeOnly,
    [switch]$ReuseEngine
)

$ErrorActionPreference = 'Stop'
$project = '/home/nvidia/yolo-jetson'
$run = 'runs/yolov5s_baseline'
$sshOptions = @('-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=20', '-o', 'ConnectTimeout=8')

function Invoke-Jetson([string]$Command) {
    ssh.exe @sshOptions jetson $Command
    if ($LASTEXITCODE -ne 0) { throw "SSH command failed ($LASTEXITCODE)." }
}

function Sync-Jetson {
    $origin = git remote get-url origin
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the Git origin.' }
    $name = ".project-$([guid]::NewGuid().ToString('N')).bundle"
    $bundle = Join-Path (Get-Location) $name
    try {
        git bundle create $bundle main
        if ($LASTEXITCODE -ne 0) { throw 'Could not bundle main.' }
        Invoke-Jetson "mkdir -p '$project'"
        scp.exe -q @sshOptions $bundle "jetson:$project/$name"
        if ($LASTEXITCODE -ne 0) { throw 'Could not upload the Git bundle.' }

        $gitCommands = @'
set -e
cd '__PROJECT__'
trap 'rm -f -- "__BUNDLE__"' EXIT
if [ ! -d .git ]; then
    git init
    git remote add origin '__ORIGIN__'
fi
git fetch './__BUNDLE__' refs/heads/main:refs/remotes/origin/main
if git show-ref --verify --quiet refs/heads/main; then
    git checkout main
    git merge --ff-only origin/main
else
    git checkout -b main --track origin/main
fi
git log -1 --oneline
'@
        $gitCommands.Replace('__PROJECT__', $project).Replace('__BUNDLE__', $name).Replace('__ORIGIN__', $origin) |
            ssh.exe @sshOptions jetson 'tr -d "\r" | bash'
        if ($LASTEXITCODE -ne 0) { throw 'Could not update main on Jetson.' }
    } finally {
        if (Test-Path -LiteralPath $bundle) { Remove-Item -LiteralPath $bundle }
    }
    if ($CodeOnly) { return }

    Invoke-Jetson "mkdir -p '$project/$run' '$project/COCO2017'"
    scp.exe -q @sshOptions "$run/yolov5s_fp32.onnx" "$run/baseline.json" "jetson:$project/$run/"
    if ($LASTEXITCODE -ne 0) { throw 'Could not upload baseline files.' }

    $datasetCheck = @'
cd '__PROJECT__/COCO2017' &&
test -s annotations/instances_val2017.json &&
test "$(find val2017 -maxdepth 1 -name '*.jpg' | wc -l)" -eq 5000
'@
    ssh.exe @sshOptions jetson ($datasetCheck.Replace('__PROJECT__', $project))
    if ($LASTEXITCODE -eq 0) {
        Write-Output 'COCO val2017 already present on Jetson.'
        return
    }

    tar.exe -cf - -C COCO2017 val2017 annotations/instances_val2017.json |
        ssh.exe @sshOptions jetson "tar -xmf - -C '$project/COCO2017'"
    if ($LASTEXITCODE -ne 0) { throw 'Could not transfer COCO val2017.' }
}

Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    switch ($Action) {
        'sync' { Sync-Jetson }
        'check' { Invoke-Jetson "cd '$project' && bash scripts/check_jetson.sh" }
        'benchmark' {
            $option = if ($ReuseEngine) { ' --reuse-engine' } else { '' }
            Invoke-Jetson "cd '$project' && bash scripts/run_jetson_fp32.sh$option"
        }
        'collect' {
            $result = "$run/jetson_fp32"
            $log = if ($ReuseEngine) { 'trtexec_reuse.log' } else { 'trtexec.log' }
            $remote = "jetson:$project/$result"
            New-Item -ItemType Directory -Force -Path $result | Out-Null
            scp.exe -q @sshOptions "$remote/$log" "$remote/sample_predictions.f32" $result
            if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve Jetson results.' }
            & .\.venv\Scripts\python.exe scripts/compare_jetson_sample.py
            if ($LASTEXITCODE -ne 0) { throw 'Jetson sample comparison failed.' }
            Remove-Item -LiteralPath "$result/sample_predictions.f32"
            scp.exe -q @sshOptions "$result/sample_coco.json" "$remote/"
            if ($LASTEXITCODE -ne 0) { throw 'Could not upload the COCO sample.' }
        }
    }
} finally {
    Pop-Location
}
