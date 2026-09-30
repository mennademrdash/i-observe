#Requires -Version 5.1
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$models = Join-Path $root 'models'
New-Item -ItemType Directory -Force -Path $models | Out-Null
$release = 'https://github.com/mennademrdash/i-observe/releases/download/models-v1'

$files = @(
  @{ Name='yolo11n-pose_widerface.pt'; Sha='766DD11CC89DFF5B13B96DB891B4E0DA9B3392ECFCD864C641C8258D496C17E8' },
  @{ Name='auraface_glintr100.onnx'; Sha='A7933EA5330113B01C9B60351D8F4C33003F145D8470AC5F0E52EE2EFFE25C60' },
  @{ Name='minifasnet_v2.onnx'; Sha='D7B3CD9BA8A7CEB13BAA8C4720902E27CA3112EFF52F926C08804AF6B6EECC7B' },
  @{ Name='sface.onnx'; Sha='0BA9FBFA01B5270C96627C4EF784DA859931E02F04419C829E83484087C34E79' },
  @{ Name='yunet.onnx'; Sha='8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4' },
  @{ Name='yolov8n.pt'; Sha='F59B3D833E2FF32E194B5BB8E08D211DC7C5BDF144B90D2C8412C47CCFC83B36' }
)

foreach ($file in $files) {
  $target = Join-Path $models $file.Name
  $valid = (Test-Path -LiteralPath $target) -and ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $file.Sha)
  if ($valid -and -not $Force) { Write-Host "[OK] $($file.Name)" -ForegroundColor Green; continue }
  Write-Host "Downloading $($file.Name)..." -ForegroundColor Cyan
  Invoke-WebRequest -Uri "$release/$($file.Name)" -OutFile $target
  $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
  if ($actual -ne $file.Sha) { Remove-Item -LiteralPath $target -Force; throw "Checksum mismatch: $($file.Name)" }
  Write-Host "[OK] $($file.Name)" -ForegroundColor Green
}

Write-Host 'Core AI models are installed and verified.' -ForegroundColor Green
Write-Host 'VideoMAE downloads from Hugging Face automatically on first video analysis.'
