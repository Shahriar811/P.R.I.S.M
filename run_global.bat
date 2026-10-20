@echo off
title P.R.I.S.M — ngrok Global Tunnel
cd /d "%~dp0"
echo.
echo Launching ngrok global public HTTPS tunnel for P.R.I.S.M (Port 7860)...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0expose_ngrok.ps1"
pause
