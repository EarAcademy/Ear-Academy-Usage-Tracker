@echo off
REM ============================================================
REM  Ear Academy — Usage Analytics & Patterns dashboard
REM  Double-click this to update the LIVE usage dashboard.
REM
REM  It does everything in one go:
REM    1. Pulls the latest from GitHub (incl. Rus's fresh AC roster)
REM    2. Rebuilds index.html from the files in daily_snapshots\
REM    3. Commits + pushes so the live site updates (~1 min later)
REM
REM  It also prints (and saves to daily_report.txt) exactly what
REM  loaded, skipped, and dropped so you can see the run went OK.
REM ============================================================

REM Run from the folder this .bat lives in, wherever that is.
cd /d "%~dp0"

REM Force UTF-8 so the report's icons don't crash the script on Windows.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Find Python: prefer the per-user install, fall back to PATH.
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo [1/3] Pulling latest from GitHub (gets Rus's newest AC roster)...
echo ----------------------------------------------------------
git pull origin main
if not %ERRORLEVEL%==0 (
  echo.
  echo   ^!^! git pull failed — resolve this before continuing, then re-run.
  echo   ^   Usually means someone else pushed at the same time, or a
  echo   ^   local file is in the way. Nothing was published.
  echo.
  pause
  exit /b 1
)

echo.
echo [2/3] Rebuilding the usage dashboard from daily_snapshots\ ...
echo ----------------------------------------------------------
"%PY%" update_dashboard.py

echo.
echo [3/3] Publishing to GitHub...
echo ----------------------------------------------------------
git add -A
git commit -m "Update usage dashboard %DATE% %TIME%"
if %ERRORLEVEL%==0 (
  git push origin main
  echo   -^> Pushed. Live site updates in about a minute.
) else (
  echo   -^> Nothing changed since last run — nothing to publish.
)

echo.
echo ============================================================
echo  Done. Read the "DAILY LOAD ^& MATCH REPORT" above:
echo    - FRESHNESS should show today's date (not STALE)
echo    - INVESTIGATE should be empty (or only known names)
echo  The same report is saved in daily_report.txt
echo ============================================================
echo.
pause
