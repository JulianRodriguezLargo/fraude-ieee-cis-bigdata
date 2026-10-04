# Descarga el dataset IEEE-CIS desde Kaggle y lo deja en data\raw (Windows / PowerShell).
#
# Requisitos (una sola vez):
#   1. py -m pip install kaggle
#   2. Token de Kaggle guardado en %USERPROFILE%\.kaggle\access_token  (ver README)
#   3. Haber aceptado las reglas en https://www.kaggle.com/c/ieee-fraud-detection/rules
#
# Uso, desde la carpeta del repositorio:
#   powershell -ExecutionPolicy Bypass -File scripts\descargar_datos.ps1

$ErrorActionPreference = "Stop"
$destino = "data\raw"
New-Item -ItemType Directory -Force -Path $destino | Out-Null

Write-Host "Descargando IEEE-CIS Fraud Detection (~124 MB)..."
py -m kaggle competitions download -c ieee-fraud-detection -p $destino

Write-Host "Descomprimiendo (~1,35 GB)..."
Expand-Archive -Path "$destino\ieee-fraud-detection.zip" -DestinationPath $destino -Force
Remove-Item "$destino\ieee-fraud-detection.zip"

Get-ChildItem $destino -Filter *.csv | Format-Table Name, @{N="MB"; E={[math]::Round($_.Length / 1MB, 1)}}
Write-Host "Listo. Los CSV estan en $destino"
