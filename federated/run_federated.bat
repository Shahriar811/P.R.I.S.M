@echo off
cd /d "%~dp0"
set FL_ROUNDS=5
set FL_CLIENTS=3
start "FL-SERVER" cmd /k python server.py
timeout /t 5 >nul
start "FL-CLIENT-0" cmd /k python client.py 0
timeout /t 1 >nul
start "FL-CLIENT-1" cmd /k python client.py 1
timeout /t 1 >nul
start "FL-CLIENT-2" cmd /k python client.py 2
timeout /t 1 >nul
echo All processes launched.
pause
