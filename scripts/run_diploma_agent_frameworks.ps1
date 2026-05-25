param(
    [string]$Framework = "all",
    [string]$Constraint = "test",
    [ValidateSet("local", "docker", "singularity", "aws")]
    [string]$Mode = "local",
    [ValidateSet("all", "classification", "regression")]
    [string]$Part = "all",
    [string[]]$Task = @(),
    [int[]]$Fold = @(),
    [ValidateSet("auto", "skip", "force", "only")]
    [string]$Setup = "auto",
    [string]$Python = "python",
    [string]$OutDir = "",
    [string[]]$Extra = @(),
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$runArgs = @(
    "scripts\run_diploma_agent_frameworks.py",
    "--framework", $Framework,
    "--constraint", $Constraint,
    "--mode", $Mode,
    "--part", $Part,
    "--setup", $Setup,
    "--python", $Python
)

if ($Task.Count -gt 0) { $runArgs += @("--task") + $Task }
if ($Fold.Count -gt 0) { $runArgs += @("--fold") + $Fold }
if ($OutDir) { $runArgs += @("--outdir", $OutDir) }
foreach ($item in $Extra) {
    if ($item) { $runArgs += @("--extra", $item) }
}
if ($ContinueOnError) { $runArgs += "--continue-on-error" }

& $Python @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "run_diploma_agent_frameworks.py failed with exit code $LASTEXITCODE"
}
