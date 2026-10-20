$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$NGROK_EXE = Join-Path $HERE "ngrok.exe"
$TOKEN_FILE = Join-Path $HERE "ngrok_token.txt"
$PORT = 7860

function Write-Section($msg) {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host " $msg" -ForegroundColor Cyan
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host ""
}

if (!(Test-Path $TOKEN_FILE)) {
    Write-Section "ngrok authtoken required"
    Write-Host "Create a file named ngrok_token.txt in this folder and paste your ngrok authtoken." -ForegroundColor Red
    exit 1
}

$TOKEN = (Get-Content $TOKEN_FILE -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($TOKEN)) {
    Write-Host "ngrok_token.txt is empty. Paste your ngrok authtoken inside it." -ForegroundColor Red
    exit 1
}

Write-Section "Registering ngrok Authtoken..."
& $NGROK_EXE config add-authtoken $TOKEN | Out-Null
Write-Host "Authtoken registered successfully." -ForegroundColor Green

Write-Section "Opening Public HTTPS Tunnel (Port $PORT)"
Write-Host "P.R.I.S.M Web Application Public Endpoints:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   - Main Dashboard :  https://<your-ngrok-url>.ngrok-free.app/" -ForegroundColor Green
Write-Host "   - Predict UI     :  https://<your-ngrok-url>.ngrok-free.app/predict-ui" -ForegroundColor Green
Write-Host "   - Federated FL   :  https://<your-ngrok-url>.ngrok-free.app/fl" -ForegroundColor Green
Write-Host "   - RL Monitor     :  https://<your-ngrok-url>.ngrok-free.app/rl" -ForegroundColor Green
Write-Host ""
Write-Host "Look below for your live ngrok public forwarding URL:" -ForegroundColor Cyan
Write-Host ""

& $NGROK_EXE http $PORT
