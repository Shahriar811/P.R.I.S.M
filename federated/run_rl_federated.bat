@echo off
REM ============================================================
REM  RL -> FL  Federated Learning launcher
REM  Train RL locally first (notebook Section 10), THEN run this.
REM  1 Flower server + 3 clients averaging Dueling RL weights
REM ============================================================
cd /d "%~dp0"
set FL_ROUNDS=5
set FL_CLIENTS=3

echo.
echo  [RL-^>FL] Starting SERVER on port 8080 ...
start "RL-FL-SERVER" cmd /k python rl_server.py

timeout /t 5 >nul

echo  [RL-^>FL] Starting CLIENT 0 ...
start "RL-FL-CLIENT-0" cmd /k python rl_client.py 0
timeout /t 1 >nul

echo  [RL-^>FL] Starting CLIENT 1 ...
start "RL-FL-CLIENT-1" cmd /k python rl_client.py 1
timeout /t 1 >nul

echo  [RL-^>FL] Starting CLIENT 2 ...
start "RL-FL-CLIENT-2" cmd /k python rl_client.py 2

echo.
echo  All 4 processes launched (federating RL weights).
echo  Watch RL-FL-SERVER for round progress.
echo  Global model -^> Saved model\FL\federated_rl_fedavg.pt
echo.
pause
