Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $root
$env:GOCACHE = Join-Path $root ".gocache"

go build -o .\tools\window_click.exe .\tools\window_click.go
Write-Host "built .\tools\window_click.exe"
