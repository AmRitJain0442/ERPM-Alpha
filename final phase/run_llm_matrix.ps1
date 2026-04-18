param(
    [string]$ApiKey = $env:OPENROUTER_API_KEY,
    [int]$TestDays = 30,
    [int]$PersonaLimit = 3,
    [string]$OutputDir = "",
    [string[]]$Models = @(
        "openrouter:google/gemini-2.5-flash",
        "openrouter:anthropic/claude-sonnet-4",
        "openrouter:openai/gpt-5-chat"
    ),
    [switch]$RichOnly,
    [switch]$LLMOnly,
    [switch]$AllowPremiumLLMModels
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "OpenRouter API key missing. Pass -ApiKey or set OPENROUTER_API_KEY."
}

$blocked = @($Models | Where-Object { $_.ToLower().Contains("claude-opus") })
if ($blocked.Count -gt 0 -and -not $AllowPremiumLLMModels) {
    throw "Blocked premium model spec(s): $($blocked -join ', '). Remove them or pass -AllowPremiumLLMModels."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run.py"
$env:OPENROUTER_API_KEY = $ApiKey

$args = @(
    $runner,
    "--test-days", $TestDays,
    "--persona-limit", $PersonaLimit
)

if ($RichOnly) {
    $args += "--rich-only"
}

if ($OutputDir) {
    $args += @("--output-dir", $OutputDir)
}

if ($AllowPremiumLLMModels) {
    $args += "--allow-premium-llm-models"
}

foreach ($model in $Models) {
    $args += @("--llm-model", $model)
}

if ($LLMOnly) {
    foreach ($model in $Models) {
        $slug = ($model -replace "[^a-zA-Z0-9]+", "_").Trim("_").ToLower()
        $args += @("--experiment", "llm_$slug")
    }
}

Write-Host "Running final-phase matrix"
Write-Host "  TestDays: $TestDays"
Write-Host "  PersonaLimit: $PersonaLimit"
Write-Host "  Models: $($Models -join ', ')"
if ($RichOnly) {
    Write-Host "  Slice: rich-only"
}
if ($LLMOnly) {
    Write-Host "  Mode: llm-only"
}

& python @args
