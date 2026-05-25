$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "manifests" | Out-Null
New-Item -ItemType Directory -Force -Path "keypoint_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path "logs" "full_training_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    $line | Tee-Object -FilePath $logPath -Append
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )

    Write-Log "START $Name"
    $exe = $Command[0]
    $args = $Command[1..($Command.Length - 1)]
    & $exe @args 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAILED $Name exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Log "DONE $Name"
}

Write-Log "Full training pipeline started"

Run-Step "prepare training keypoints" @(
    "python", "-m", "model.prepare_keypoints",
    "--split-root", "1.Training",
    "--cache-root", "keypoint_cache",
    "--manifest-out", "manifests/train.json",
    "--task-model-path", "model/assets/holistic_landmarker.task",
    "--max-frames", "64",
    "--skip-existing"
)

Run-Step "prepare validation keypoints" @(
    "python", "-m", "model.prepare_keypoints",
    "--split-root", "2.Validation",
    "--cache-root", "keypoint_cache",
    "--manifest-out", "manifests/val.json",
    "--task-model-path", "model/assets/holistic_landmarker.task",
    "--max-frames", "64",
    "--skip-existing"
)

Run-Step "train word classifier" @(
    "python", "-m", "model.train_word_classifier",
    "--train-manifest", "manifests/train.json",
    "--val-manifest", "manifests/val.json",
    "--output-dir", "checkpoints/word_classifier",
    "--epochs", "20",
    "--batch-size", "16",
    "--max-frames", "64"
)

Write-Log "Full training pipeline finished"
