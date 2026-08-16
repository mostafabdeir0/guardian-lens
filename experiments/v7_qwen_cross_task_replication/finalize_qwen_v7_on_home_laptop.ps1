param(
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

$BRANCH = "v7-qwen-cross-task-replication"
$FINAL_TAG = "guardian-lens-v7-qwen-cross-task-replication-final-2026-08-16"

$EXPECTED_RAW_SHA = "ce7fea1762084810ba571a26b91816e4834decd1e404478188fe4dd1239a36f4"
$EXPECTED_ANALYZER_SHA = "df00311a6f4b2c177b8e56d3824649f4c08e5b57153aa5ff952c3aa86d5f8b0b"
$EXPECTED_PRED_SHA = "c89c3e1ed5b1434baf66e450bec2e0a2fff310a816c18fff75a715aa485c61a4"

$ROOT = (
    Resolve-Path (
        Join-Path $PSScriptRoot "..\.."
    )
).Path

Set-Location $ROOT

$EXP = Join-Path $ROOT "experiments\v7_qwen_cross_task_replication"

$RAW = Join-Path $EXP "outputs\qwen_v7_raw.jsonl"
$RAW_FREEZE = Join-Path $EXP "QWEN_V7_RAW_FREEZE.json"

$ANALYZER = Join-Path $EXP "analyze_qwen_v7_cross_task.py"
$ANALYZER_FREEZE = Join-Path $EXP "QWEN_V7_ANALYZER_FREEZE.json"

$PREDICTIONS = Join-Path $ROOT "experiments\v7_cross_task_validity\taskA_predictions.csv"

$ANALYSIS = Join-Path $EXP "analysis"
$RESULTS_FREEZE = Join-Path $EXP "QWEN_V7_RESULTS_FREEZE.json"

function Get-Sha256 {
    param([string]$Path)

    return (
        Get-FileHash $Path -Algorithm SHA256
    ).Hash.ToLower()
}

function Assert-Equal {
    param(
        [string]$Actual,
        [string]$Expected,
        [string]$Label
    )

    if ($Actual -ne $Expected) {
        throw (
            "STOP: {0} mismatch.`nExpected: {1}`nActual:   {2}" -f
            $Label,
            $Expected,
            $Actual
        )
    }
}

Write-Host "`n========================================"
Write-Host "QWEN V7 FINAL CONFIRMATORY ANALYSIS"
Write-Host "========================================"

Write-Host "`n===== 1. CLEAN REPOSITORY ====="

$currentBranch = (
    git branch --show-current
).Trim()

if ($currentBranch -ne $BRANCH) {
    throw "STOP: expected branch $BRANCH; found $currentBranch"
}

$dirty = @(
    git status --porcelain
)

if ($dirty.Count -ne 0) {
    git status
    throw "STOP: repository is not clean."
}

Write-Host "Branch:" $currentBranch
Write-Host "Working tree: clean"

Write-Host "`n===== 2. PRE-ANALYSIS FREEZE HISTORY ====="

git merge-base --is-ancestor bc9d9a2 HEAD

if ($LASTEXITCODE -ne 0) {
    throw "STOP: raw-freeze commit bc9d9a2 is not an ancestor of HEAD."
}

git merge-base --is-ancestor 4f070da HEAD

if ($LASTEXITCODE -ne 0) {
    throw "STOP: analyzer-freeze commit 4f070da is not an ancestor of HEAD."
}

$rawFreezeCommit = (
    git rev-parse bc9d9a2
).Trim()

$analyzerFreezeCommit = (
    git rev-parse 4f070da
).Trim()

Write-Host "Raw freeze:" $rawFreezeCommit
Write-Host "Analyzer freeze:" $analyzerFreezeCommit

Write-Host "`n===== 3. FROZEN INPUT HASHES ====="

$rawHash = Get-Sha256 $RAW
$analyzerHash = Get-Sha256 $ANALYZER
$predHash = Get-Sha256 $PREDICTIONS

Write-Host "Raw:" $rawHash
Write-Host "Analyzer:" $analyzerHash
Write-Host "Task-A predictions:" $predHash

Assert-Equal $rawHash $EXPECTED_RAW_SHA "Qwen raw SHA256"
Assert-Equal $analyzerHash $EXPECTED_ANALYZER_SHA "Qwen analyzer SHA256"
Assert-Equal $predHash $EXPECTED_PRED_SHA "Task-A prediction SHA256"

$analyzerFreeze = (
    Get-Content $ANALYZER_FREEZE -Raw |
    ConvertFrom-Json
)

if (
    $analyzerFreeze.status -ne
    "FROZEN_BEFORE_ANY_QWEN_V7_ANALYSIS_EXECUTION"
) {
    throw "STOP: unexpected analyzer-freeze status."
}

Assert-Equal `
    $analyzerFreeze.raw_sha256 `
    $EXPECTED_RAW_SHA `
    "Analyzer-freeze raw SHA256"

Assert-Equal `
    $analyzerFreeze.analyzer_sha256 `
    $EXPECTED_ANALYZER_SHA `
    "Analyzer-freeze analyzer SHA256"

Assert-Equal `
    $analyzerFreeze.taskA_predictions_sha256 `
    $EXPECTED_PRED_SHA `
    "Analyzer-freeze Task-A SHA256"

Write-Host "PASS: frozen provenance verified."

Write-Host "`n===== 4. NO PRIOR ANALYSIS ====="

if (Test-Path $ANALYSIS) {
    throw "STOP: analysis directory already exists."
}

if (Test-Path $RESULTS_FREEZE) {
    throw "STOP: results freeze already exists."
}

$existingTag = git tag --list $FINAL_TAG

if ($existingTag) {
    throw "STOP: final tag already exists locally."
}

$remoteTag = git ls-remote --tags origin "refs/tags/$FINAL_TAG"

if ($LASTEXITCODE -ne 0) {
    throw "STOP: could not query remote tags."
}

if ($remoteTag) {
    throw "STOP: final tag already exists on origin."
}

Write-Host "PASS: no prior Qwen analysis/results freeze."

Write-Host "`n===== 5. RUN FROZEN ANALYZER ====="
Write-Host "Analyzer output will remain hidden until results are committed."

$analysisLog = Join-Path (
    [System.IO.Path]::GetTempPath()
) (
    "guardian_lens_qwen_v7_" +
    [guid]::NewGuid().ToString("N") +
    ".log"
)

try {
    & python $ANALYZER *> $analysisLog

    $analysisExit = $LASTEXITCODE

    if ($analysisExit -ne 0) {
        Write-Host "`nANALYZER FAILED:"
        Get-Content $analysisLog
        throw "STOP: frozen analyzer returned nonzero exit code."
    }

    if (-not (Test-Path $ANALYSIS)) {
        throw "STOP: analyzer completed but analysis directory is absent."
    }

    $analysisFiles = @(
        Get-ChildItem $ANALYSIS -File |
        Sort-Object Name
    )

    if ($analysisFiles.Count -ne 9) {
        throw (
            "STOP: expected exactly 9 analysis output files; found " +
            $analysisFiles.Count
        )
    }

    $requiredOutputs = @(
        "analysis_metadata.json",
        "confirmatory_results.csv",
        "descriptive_summary.json",
        "taskB_cell_means.csv",
        "pv1_scene_effects.csv",
        "pv2_scene_effects.csv",
        "pv3_scene_effects.csv"
    )

    foreach ($name in $requiredOutputs) {
        $path = Join-Path $ANALYSIS $name

        if (-not (Test-Path $path)) {
            throw "STOP: required analysis output missing: $name"
        }

        if ((Get-Item $path).Length -le 0) {
            throw "STOP: empty analysis output: $name"
        }
    }

    Write-Host "PASS: analyzer completed and 9 output files were created."

    Write-Host "`n===== 6. HASH ALL ANALYSIS OUTPUTS ====="

    $outputRecords = @()

    foreach ($file in $analysisFiles) {
        $relative = (
            $file.FullName.Substring(
                $ROOT.Length + 1
            )
        ).Replace("\", "/")

        $outputRecords += [ordered]@{
            path = $relative
            sha256 = Get-Sha256 $file.FullName
            bytes = $file.Length
        }
    }

    $preAnalysisHandoffCommit = (
        git rev-parse HEAD
    ).Trim()

    $rawFreezeHash = Get-Sha256 $RAW_FREEZE
    $analyzerFreezeHash = Get-Sha256 $ANALYZER_FREEZE
    $finalizerHash = Get-Sha256 $PSCommandPath

    $resultsFreezeObject = [ordered]@{
        status = "FROZEN_IMMEDIATELY_AFTER_QWEN_V7_CONFIRMATORY_ANALYSIS"

        preanalysis_handoff_commit = $preAnalysisHandoffCommit
        raw_freeze_commit = $rawFreezeCommit
        analyzer_freeze_commit = $analyzerFreezeCommit

        raw_sha256 = $rawHash
        raw_freeze_sha256 = $rawFreezeHash

        analyzer_sha256 = $analyzerHash
        analyzer_freeze_sha256 = $analyzerFreezeHash

        taskA_predictions_sha256 = $predHash

        finalizer_path = "experiments/v7_qwen_cross_task_replication/finalize_qwen_v7_on_home_laptop.ps1"
        finalizer_sha256 = $finalizerHash

        analyzer_exit_code = 0

        output_file_count = $analysisFiles.Count
        outputs = $outputRecords

        bootstrap_replicates = 20000
        bootstrap_seed = 20260816
        alpha = 0.05

        confirmatory_family = @(
            "PV1",
            "PV2",
            "PV3"
        )

        multiple_testing = "Holm"
        inferential_unit = "scene"

        analyzer_stdout_revealed_before_results_freeze_commit = $false
        results_inspected_before_results_freeze_commit = $false

        gemini_qwen_pooling = "NONE"
    }

    $freezeJson = (
        $resultsFreezeObject |
        ConvertTo-Json -Depth 8
    )

    $freezeJson = $freezeJson.Replace(
        "`r`n",
        "`n"
    )

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)

    [System.IO.File]::WriteAllText(
        $RESULTS_FREEZE,
        $freezeJson + "`n",
        $utf8NoBom
    )

    Write-Host "`n===== 7. STAGE + BYTE-VERIFY RESULTS ====="

    git add `
        experiments/v7_qwen_cross_task_replication/analysis `
        experiments/v7_qwen_cross_task_replication/QWEN_V7_RESULTS_FREEZE.json

    if ($LASTEXITCODE -ne 0) {
        throw "STOP: git add failed."
    }

    git diff --cached --check

    if ($LASTEXITCODE -ne 0) {
        throw "STOP: staged diff check failed."
    }

    foreach ($record in $outputRecords) {
        $relative = $record.path

        $indexHash = (
            python -c "import sys,subprocess,hashlib; b=subprocess.check_output(['git','show',':'+sys.argv[1]]); print(hashlib.sha256(b).hexdigest())" $relative
        ).Trim()

        if ($indexHash -ne $record.sha256) {
            throw (
                "STOP: staged bytes differ for " +
                $relative
            )
        }
    }

    Write-Host "PASS: staged analysis bytes match frozen hashes."

    Write-Host "`n===== 8. COMMIT RESULTS BEFORE REVEAL ====="

    git commit -m "Record Qwen V7 cross-task predictive validity results"

    if ($LASTEXITCODE -ne 0) {
        throw "STOP: results commit failed."
    }

    $resultsCommit = (
        git rev-parse HEAD
    ).Trim()

    git tag -a `
        $FINAL_TAG `
        -m "Guardian Lens Qwen V7 cross-task replication final"

    if ($LASTEXITCODE -ne 0) {
        throw "STOP: final tag creation failed."
    }

    Write-Host "Results commit:" $resultsCommit
    Write-Host "Final tag:" $FINAL_TAG

    Write-Host "`n========================================"
    Write-Host "RESULTS ARE NOW FROZEN IN GIT"
    Write-Host "========================================"

    Write-Host "`n===== CONFIRMATORY ANALYZER OUTPUT ====="
    Get-Content $analysisLog

    Write-Host "`n===== 9. RESULTS FREEZE SHA256 ====="
    Write-Host (Get-Sha256 $RESULTS_FREEZE)

    if (-not $NoPush) {
        Write-Host "`n===== 10. PUSH FINAL BRANCH + TAG ====="

        git push origin $BRANCH

        if ($LASTEXITCODE -ne 0) {
            throw "Results are safely committed locally, but branch push failed."
        }

        git push origin $FINAL_TAG

        if ($LASTEXITCODE -ne 0) {
            throw "Branch pushed, but final-tag push failed."
        }

        Write-Host "PASS: final branch and tag pushed."
    }
    else {
        Write-Host "`nPush skipped because -NoPush was supplied."
    }

    Write-Host "`n===== FINAL REPOSITORY STATE ====="
    git status
    git log -5 --oneline --decorate

    Write-Host "`nQWEN V7 FINALIZATION COMPLETE."
}
finally {
    if (Test-Path $analysisLog) {
        Remove-Item $analysisLog -Force
    }
}
