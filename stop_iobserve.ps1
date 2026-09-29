#Requires -Version 5.1
<#
  Stops ONLY I-observe resources.
  Leaves the native PostgreSQL 18 service, the office-security-app stack and every
  other container/project untouched.
#>
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$port = 8000

function Info($m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [OK]   $m" -ForegroundColor Green }

Write-Host ""
Write-Host "  Stopping I-observe ..." -ForegroundColor White

# 1. the web console
$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  $busy | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
  Ok "web console on port $port stopped"
} else {
  Info "no web console was listening on $port"
}

# 2. the S3-compatible evidence endpoint started by start_iobserve.ps1
$moto = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*moto.server*' }
if ($moto) {
  $moto | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Ok "evidence object storage stopped"
} else {
  Info "no moto evidence server was running"
}

# 3. only this project's compose services (PostgreSQL). MinIO is opt-in, so
#    --profile is not passed and MinIO is never touched.
if (Get-Command docker -ErrorAction SilentlyContinue) {
  docker info --format '{{.ServerVersion}}' *> $null
  if ($LASTEXITCODE -eq 0) {
    docker compose down 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Ok "PostgreSQL stopped (project data volumes kept)" }
    else { Info "compose down reported a problem; leaving containers as they are" }
  } else {
    Info "Docker engine not running, nothing to stop"
  }
} else {
  Info "Docker not found on PATH, nothing to stop"
}

Ok "done - office-security-app, the native PostgreSQL service and other projects were not touched"
Write-Host ""
