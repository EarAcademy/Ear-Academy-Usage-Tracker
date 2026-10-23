@echo off
REM ============================================================
REM  Ear Academy — Update ALL 3 dashboards (Windows daily driver)
REM  Windows equivalent of update_all_dashboards.sh.
REM
REM  Runs, in order:
REM    1. Sales    -> pipeline_data.json + paying_schools.json (AC roster)
REM    2. Usage    -> index.html   (reads the fresh roster)
REM    3. Velocity -> velocity_data.json
REM  then a single git add + commit + push.
REM
REM  Each step is independent; a failure in one does not stop the others.
REM  After it finishes, hard-refresh the live dashboards (Ctrl+Shift+R):
REM    https://earacademy.github.io/Ear-Academy-Usage-Tracker/index.html
REM    https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html
REM    https://earacademy.github.io/Ear-Academy-Usage-Tracker/pipeline_velocity.html
REM ============================================================

cd /d "%~dp0"

REM UTF-8 so the scripts' status icons don't crash on the Windows console codepage.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Prefer the real per-user Python install, fall back to PATH.
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

set "SALES_OK=0"
set "USAGE_OK=0"
set "VELOCITY_OK=0"

echo.
echo ==========================================================
echo   Ear Academy - Updating all 3 dashboards
echo ==========================================================

echo.
echo [0/3] Pulling latest from GitHub...
echo ----------------------------------------------------------
git pull origin main
if not %ERRORLEVEL%==0 (
  echo.
  echo   ^!^! git pull failed — resolve this before continuing, then re-run.
  echo   ^   Nothing was published.
  echo.
  pause
  exit /b 1
)

echo.
echo [1/3] Sales dashboard (pulls live data from ActiveCampaign)
echo ----------------------------------------------------------
"%PY%" update_sales_dashboard.py --no-push
if %ERRORLEVEL%==0 (set "SALES_OK=1") else (echo   ^!^! Sales step failed)

echo.
echo [2/3] Usage dashboard (rebuilds index.html from snapshots + roster)
echo ----------------------------------------------------------
"%PY%" update_dashboard.py
if %ERRORLEVEL%==0 (set "USAGE_OK=1") else (echo   ^!^! Usage step failed)

echo.
echo [3/3] Velocity dashboard (pulls live data from ActiveCampaign)
echo ----------------------------------------------------------
"%PY%" update_velocity.py --no-push
if %ERRORLEVEL%==0 (set "VELOCITY_OK=1") else (echo   ^!^! Velocity step failed)

echo.
echo ==========================================================
echo   Publishing to GitHub (single commit)
echo ==========================================================
git add -A
git commit -m "Update dashboards [usage+sales+velocity] %DATE% %TIME%"
if %ERRORLEVEL%==0 (
  git push origin main
) else (
  echo   Nothing to commit, or commit failed - nothing pushed.
)

echo.
echo ==========================================================
echo   SUMMARY   Sales=%SALES_OK%  Usage=%USAGE_OK%  Velocity=%VELOCITY_OK%   (1 = ok, 0 = failed)
echo   Read the DAILY LOAD ^& MATCH REPORT above for the usage dashboard.
echo ==========================================================
echo.
pause
