#Requires -Version 5.1
<#
  I-observe launcher.
  Starts only what I-observe needs: PostgreSQL (Docker), the S3-compatible
  evidence endpoint, and the web console. MinIO is optional and excluded.
  Never touches the machine's native PostgreSQL or the office-security-app stack.
#>
# NOTE: $ErrorActionPreference is deliberately 'Continue'. The Docker CLI writes progress
# to stderr even on success ("Container ... Running"), and 'Stop' would turn that harmless
# chatter into a terminating error. Every step below checks $LASTEXITCODE explicitly.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = Join-Path $root '.venv\Scripts\python.exe'
$port = 8000
$url = "http://localhost:$port"

function Info($m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [OK]   $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "  [FAIL] $m" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  I-Observe  -  Video Intelligence Console" -ForegroundColor White
Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
Write-Host "  repository: $root" -ForegroundColor DarkGray
Write-Host ""

# ---- 1. python ----
if (-not (Test-Path $py)) { Die "Virtual environment missing. Run: py -m venv .venv ; .\.venv\Scripts\python.exe -m pip install -r requirements.txt" }
Info "using virtual environment"

# ---- 2. config ----
if (-not (Test-Path (Join-Path $root '.env'))) { Die "Missing .env (secret configuration). It must define DATABASE_URL and the storage variables." }
# ---- 2b. VLM provider (key comes from THIS session's environment, never from disk) ----
# Set the key in this terminal before running:  $env:OPENROUTER_API_KEY = "..."
$present = [bool]$env:OPENROUTER_API_KEY
if ($present) { Info "OpenRouter key present in this session (OPENROUTER_API_KEY)" }
else { Info "no VLM key in this session - set OPENROUTER_API_KEY then re-run this script" }
if (-not $env:VLM_PROVIDER) { $env:VLM_PROVIDER = 'openrouter' }
if (-not $env:OPENROUTER_BASE_URL) { $env:OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1' }
if ($env:VLM_PROVIDER -eq 'openrouter' -and -not $env:VLM_MODEL) { $env:VLM_MODEL = 'google/gemini-3.6-flash' }
if (-not $env:VLM_BASE_URL) { $env:VLM_BASE_URL = $env:OPENROUTER_BASE_URL }
Info "VLM_PROVIDER=$env:VLM_PROVIDER  model=$env:VLM_MODEL"

# ---- 3. docker services (required only) ----
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Die "Docker not found on PATH." }
docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) { Die "Docker engine is not running. Start Docker Desktop and retry." }

Info "starting required services (postgres, redis) ..."
docker compose up -d postgres redis 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Die "docker compose failed to start PostgreSQL/Redis." }Ok "PostgreSQL and Redis requested (MinIO is optional and excluded)"

# ---- 4. wait for postgres ----
$pgPort = 5433
$pgReady = $false
for ($i = 0; $i -lt 30; $i++) {
  $s = docker ps --filter "name=iobserve-postgres" --format '{{.Status}}' 2>$null
  if ($s -match 'healthy') { $pgReady = $true; break }
  Start-Sleep -Seconds 2
}
if ($pgReady) { Ok "PostgreSQL healthy (localhost:$pgPort)" } else { Warn "PostgreSQL not healthy yet - the app will still start and report OFFLINE on the dashboard." }

# ---- 6. S3-compatible evidence endpoint (optional) ----
$s3Up = $false
try { if ((Invoke-WebRequest -Uri 'http://localhost:9000/' -UseBasicParsing -TimeoutSec 3).StatusCode -lt 500) { $s3Up = $true } } catch {}
if (-not $s3Up) {
  Info "starting S3-compatible evidence endpoint on :9000 ..."
  Start-Process -FilePath $py -ArgumentList '-m', 'moto.server', '-p', '9000' -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $root 'tools\moto.log') -RedirectStandardError (Join-Path $root 'tools\moto.err')
  for ($i = 0; $i -lt 20; $i++) {
    try { if ((Invoke-WebRequest -Uri 'http://localhost:9000/' -UseBasicParsing -TimeoutSec 3).StatusCode -lt 500) { $s3Up = $true; break } } catch {}
    Start-Sleep -Seconds 1
  }
}
if ($s3Up) { Ok "evidence object storage on :9000" } else { Warn "evidence object storage unavailable - evidence stays on local disk and the dashboard shows OFFLINE." }

# ---- 7. free the port ----
$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  Warn "port $port is in use; stopping the previous I-observe server"
  $busy | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

# ---- 8. start the app ----
Info "starting web console on $url"
Start-Process -FilePath $py -ArgumentList '-m', 'uvicorn', 'query_api:app', '--host', '127.0.0.1', '--port', "$port" `
  -WindowStyle Hidden -WorkingDirectory $root `
  -RedirectStandardOutput (Join-Path $root 'tools\server.log') -RedirectStandardError (Join-Path $root 'tools\server.err')

$up = $false
for ($i = 0; $i -lt 40; $i++) {
  try { if ((Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $up = $true; break } } catch {}
  Start-Sleep -Seconds 1
}
if (-not $up) {
  Die "The web console did not become healthy. Check tools\server.err"
}
Ok "web console is healthy"

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host "   OPEN THIS URL:   $url" -ForegroundColor Green
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Stop everything:  .\stop_iobserve.ps1" -ForegroundColor DarkGray
Write-Host "  Logs:             tools\server.log, tools\server.err" -ForegroundColor DarkGray
Write-Host ""
