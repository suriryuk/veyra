[CmdletBinding()]
param(
    [switch]$Force
)

$envPath = Join-Path $PSScriptRoot ".env"

if ((Test-Path -LiteralPath $envPath) -and -not $Force) {
    Write-Host ".env already exists. Use -Force to replace it."
    exit 0
}

$secretBytes = New-Object byte[] 32
$random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $random.GetBytes($secretBytes)
}
finally {
    $random.Dispose()
}

$secret = ([System.BitConverter]::ToString($secretBytes)).Replace("-", "").ToLowerInvariant()
$lines = @(
    "SEARXNG_HOST=127.0.0.1"
    "SEARXNG_PORT=8888"
    "SEARXNG_VERSION=2026.8.20-8d3dd0cd4"
    "VALKEY_VERSION=9.1.2-alpine3.24"
    "SEARXNG_SECRET=$secret"
)
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($envPath, $lines, $utf8WithoutBom)

Write-Host "Created $envPath with a random SearXNG secret."
