$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$NumShards = 8
$Epochs = 20
$BatchSize = 16
$MaxFrames = 64

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "manifests" | Out-Null
New-Item -ItemType Directory -Force -Path "keypoint_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$mainLog = Join-Path "logs" "full_training_parallel_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    $line | Tee-Object -FilePath $mainLog -Append
}

function Start-Shards {
    param(
        [string]$SplitRoot,
        [string]$ManifestPrefix
    )

    Write-Log "START parallel keypoint extraction: $SplitRoot with $NumShards shards"
    $processes = @()
    for ($i = 0; $i -lt $NumShards; $i++) {
        $outLog = Join-Path "logs" "$ManifestPrefix`_shard_$i`_$timestamp.out.log"
        $errLog = Join-Path "logs" "$ManifestPrefix`_shard_$i`_$timestamp.err.log"
        $manifest = "manifests/$ManifestPrefix`_shard_$i.json"
        $args = @(
            "-m", "model.prepare_keypoints",
            "--split-root", $SplitRoot,
            "--cache-root", "keypoint_cache",
            "--manifest-out", $manifest,
            "--task-model-path", "model/assets/holistic_landmarker.task",
            "--max-frames", "$MaxFrames",
            "--num-shards", "$NumShards",
            "--shard-index", "$i",
            "--skip-existing"
        )
        $p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $Root -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
        Write-Log "  shard $i pid=$($p.Id) manifest=$manifest"
        $processes += $p
    }

    Wait-Process -Id ($processes.Id)
    foreach ($p in $processes) {
        $p.Refresh()
        Write-Log "  shard pid=$($p.Id) exit=$($p.ExitCode)"
        if ($p.ExitCode -ne 0) {
            throw "Shard failed: pid=$($p.Id), exit=$($p.ExitCode)"
        }
    }

    $inputs = @()
    for ($i = 0; $i -lt $NumShards; $i++) {
        $inputs += "manifests/$ManifestPrefix`_shard_$i.json"
    }
    & python -m model.combine_manifests --inputs $inputs --output "manifests/$ManifestPrefix.json" 2>&1 | Tee-Object -FilePath $mainLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Manifest combine failed for $ManifestPrefix"
    }
    Write-Log "DONE parallel keypoint extraction: $SplitRoot"
}

Write-Log "Full parallel training pipeline started"
Start-Shards -SplitRoot "1.Training" -ManifestPrefix "train"
Start-Shards -SplitRoot "2.Validation" -ManifestPrefix "val"

Write-Log "START GPU training"
& python -m model.train_word_classifier `
    --train-manifest "manifests/train.json" `
    --val-manifest "manifests/val.json" `
    --output-dir "checkpoints/word_classifier" `
    --epochs "$Epochs" `
    --batch-size "$BatchSize" `
    --max-frames "$MaxFrames" 2>&1 | Tee-Object -FilePath $mainLog -Append
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit=$LASTEXITCODE"
}
Write-Log "Full parallel training pipeline finished"
