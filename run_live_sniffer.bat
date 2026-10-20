@echo off
title P.R.I.S.M — Live Real Network Packet Sniffer
cd /d "%~dp0"
echo.
echo ===============================================================
echo   P.R.I.S.M — Real-Life Real-Time Network Traffic Sniffer
echo ===============================================================
echo.
echo Installing/checking 'scapy' dependency...
pip install scapy
echo.
echo Starting live packet sniffer...
python live_sniffer.py
pause
