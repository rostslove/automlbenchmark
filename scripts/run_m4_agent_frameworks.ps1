param(
    [string[]]$Groups = @("Yearly", "Monthly", "Quarterly", "Daily"),
    [int]$NPerGroup = 0,
    [int]$WindowLength = 50,
    [switch]$NoStandardize,
    [int]$Folds = 2,
    [double]$TestSize = 0.2,
    [int]$Seed = 42,
    [string]$OutputDir = "data/m4_frequency_classification",
    [string]$Framework = "all",
    [string]$Constraint = "test",
    [ValidateSet("local", "docker", "singularity", "aws")]
    [string]$Mode = "local",
    [int[]]$Fold = @(0),
    [ValidateSet("auto", "skip", "force", "only")]
    [string]$Setup = "auto",
    [string]$Python = "python",
    [string]$ResultsDir = "",
    [string[]]$Extra = @(),
    [switch]$Ollama,
    [string]$OllamaUrl = "http://127.0.0.1:11434/v1",
    [string]$OllamaModel = "gpt-4o-mini",
    [switch]$ContinueOnError,
    [switch]$PrepareOnly,
    [switch]$Force,
    [int]$ChunkRows = 512
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$runArgs = @(
    "scripts\run_m4_agent_frameworks.py",
    "--groups"
) + $Groups + @(
    "--n-per-group", $NPerGroup,
    "--window-length", $WindowLength,
    "--folds", $Folds,
    "--test-size", $TestSize,
    "--seed", $Seed,
    "--output-dir", $OutputDir,
    "--framework", $Framework,
    "--constraint", $Constraint,
    "--mode", $Mode,
    "--setup", $Setup,
    "--python", $Python,
    "--chunk-rows", $ChunkRows
)

if ($NoStandardize) { $runArgs += "--no-standardize" }
if ($Fold.Count -gt 0) { $runArgs += @("--fold") + $Fold }
if ($ResultsDir) { $runArgs += @("--results-dir", $ResultsDir) }
if ($Ollama) { $runArgs += @("--ollama", "--ollama-url", $OllamaUrl, "--ollama-model", $OllamaModel) }
foreach ($item in $Extra) {
    if ($item) { $runArgs += @("--extra", $item) }
}
if ($ContinueOnError) { $runArgs += "--continue-on-error" }
if ($PrepareOnly) { $runArgs += "--prepare-only" }
if ($Force) { $runArgs += "--force" }

& $Python @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "run_m4_agent_frameworks.py failed with exit code $LASTEXITCODE"
}
