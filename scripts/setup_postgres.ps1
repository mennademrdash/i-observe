# One-time provisioning of the observe role and database.
#
# Credentials are read from the project .env file and are never echoed: every psql call
# receives the superuser password through the PGPASSWORD environment variable and all
# output is filtered to a plain status line.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root '.env'
if (-not (Test-Path $envFile)) { throw "Missing $envFile (secret configuration)." }

function Get-EnvValue([string]$Name) {
    foreach ($line in Get-Content $envFile) {
        $t = $line.Trim()
        if ($t -and -not $t.StartsWith('#') -and $t.Contains('=')) {
            $k, $v = $t.Split('=', 2)
            if ($k.Trim() -eq $Name) { return $v.Trim().Trim('"').Trim("'") }
        }
    }
    return $null
}

$superuser = Get-EnvValue 'PG_SUPERUSER'
$superPw   = Get-EnvValue 'PG_SUPERUSER_PASSWORD'
$dsn       = Get-EnvValue 'DATABASE_URL'
if (-not $superPw) { throw 'PG_SUPERUSER_PASSWORD not set in .env' }
if (-not $dsn)     { throw 'DATABASE_URL not set in .env' }

# Parse role/password/db out of the application DSN without printing it.
$m = [regex]::Match($dsn, '^postgresql://([^:]+):([^@]+)@[^/]+/(\w+)$')
if (-not $m.Success) { throw 'DATABASE_URL is not in the expected postgresql://user:pass@host/db form' }
$role     = $m.Groups[1].Value
$rolePw   = $m.Groups[2].Value
$dbname   = $m.Groups[3].Value

$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
if (-not (Test-Path $psql)) { throw "psql not found at $psql" }

$env:PGPASSWORD = $superPw

function Invoke-Psql([string]$Sql, [string]$Db = 'postgres') {
    & $psql -U $superuser -h 127.0.0.1 -p 5432 -d $Db -v ON_ERROR_STOP=1 -tAc $Sql 2>&1
}

# 1. role
$exists = (Invoke-Psql "SELECT 1 FROM pg_roles WHERE rolname='$role'")
if ($exists -notmatch '1') {
    [void](Invoke-Psql "CREATE ROLE $role WITH LOGIN PASSWORD ''")
    # Set the password via a separate statement so the literal never lands in shell history
    # through the function body above.
    $setPw = "ALTER ROLE $role WITH LOGIN PASSWORD '$rolePw'"
    [void](& $psql -U $superuser -h 127.0.0.1 -p 5432 -d postgres -v ON_ERROR_STOP=1 -tAc $setPw 2>&1)
    Write-Output "role '$role' created"
} else {
    $setPw = "ALTER ROLE $role WITH LOGIN PASSWORD '$rolePw'"
    [void](& $psql -U $superuser -h 127.0.0.1 -p 5432 -d postgres -v ON_ERROR_STOP=1 -tAc $setPw 2>&1)
    Write-Output "role '$role' already exists (password refreshed)"
}

# 2. database
$dbExists = (Invoke-Psql "SELECT 1 FROM pg_database WHERE datname='$dbname'")
if ($dbExists -notmatch '1') {
    [void](Invoke-Psql "CREATE DATABASE $dbname OWNER $role")
    Write-Output "database '$dbname' created"
} else {
    Write-Output "database '$dbname' already exists"
}

# 3. privileges for the app to create its events table
[void](Invoke-Psql "GRANT ALL PRIVILEGES ON DATABASE $dbname TO $role")
[void](& $psql -U $superuser -h 127.0.0.1 -p 5432 -d $dbname -v ON_ERROR_STOP=1 -tAc "GRANT ALL ON SCHEMA public TO $role" 2>&1)

# 4. verify the APPLICATION credential works end to end (not the superuser one)
$env:PGPASSWORD = $rolePw
$who = (& $psql -U $role -h 127.0.0.1 -p 5432 -d $dbname -tAc 'SELECT current_user' 2>&1)
if ($who -match $role) { Write-Output "application login OK as '$role' on '$dbname'" }
else { throw "application login FAILED for '$role': $who" }
