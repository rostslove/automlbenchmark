param(
    [string]$Framework = "AutoGluon",
    [string]$Constraint = "test",
    [ValidateSet("local", "docker", "singularity", "aws")]
    [string]$Mode = "docker",
    [ValidateSet("all", "classification", "regression")]
    [string]$Part = "all",
    [ValidateSet("auto", "skip", "force", "only")]
    [string]$Setup = "auto"
)

$ErrorActionPreference = "Stop"
$defaultFrameworks = @(
    "AutoGluon",
    "flaml",
    "H2OAutoML",
    "lightautoml",
    "mljarsupervised",
    "TPOT",
    "RandomForest"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$benchmark = "diploma_mixed"
$classificationTasks = @(
    "kc2_binary_classification",
    "iris_multiclass_classification",
    "credit_g_binary_classification"
)
$regressionTasks = @(
    "cholesterol_regression",
    "autoMpg_regression",
    "kin8nm_regression"
)

$frameworks = @($Framework)
if ($Framework -eq "all") {
    $frameworks = $defaultFrameworks
}
elseif ($Framework.Contains(",")) {
    $frameworks = $Framework.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$failedFrameworks = @()
foreach ($fw in $frameworks) {
    Write-Host ""
    Write-Host "===== Running $fw on $benchmark ($Part, $Constraint, $Mode, setup=$Setup) ====="
    $runArgs = @("runbenchmark.py", $fw, $benchmark, $Constraint, "-m", $Mode, "-s", $Setup)
    if ($Part -eq "classification") {
        $runArgs += @("-t") + $classificationTasks
    }
    elseif ($Part -eq "regression") {
        $runArgs += @("-t") + $regressionTasks
    }

    python @runArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "===== $fw`: OK ====="
    }
    else {
        Write-Warning "===== $fw`: FAILED ====="
        $failedFrameworks += $fw
    }
}

if ($failedFrameworks.Count -gt 0) {
    throw "Failed frameworks: $($failedFrameworks -join ', ')"
}
