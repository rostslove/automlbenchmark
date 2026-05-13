param(
    [string]$Framework = "AutoGluon",
    [string]$Constraint = "test",
    [ValidateSet("local", "docker", "singularity", "aws")]
    [string]$Mode = "docker",
    [ValidateSet("all", "classification", "regression")]
    [string]$Part = "all"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$benchmark = "diploma_mixed"
$classificationTasks = @(
    "kc2_binary_classification",
    "iris_multiclass_classification"
)
$regressionTasks = @(
    "cholesterol_regression",
    "autoMpg_regression"
)

$argsList = @("runbenchmark.py", $Framework, $benchmark, $Constraint, "-m", $Mode)

if ($Part -eq "classification") {
    $argsList += @("-t") + $classificationTasks
}
elseif ($Part -eq "regression") {
    $argsList += @("-t") + $regressionTasks
}

python @argsList
