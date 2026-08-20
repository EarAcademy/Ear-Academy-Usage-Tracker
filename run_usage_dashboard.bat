@echo off
REM ============================================================
REM  Ear Academy — Usage Analytics & Patterns dashboard
REM  Double-click this to rebuild index.html from the snapshots.
REM  It prints (and saves to daily_report.txt) exactly what was
REM  loaded, skipped, and dropped so you can see every run went OK.
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
echo Rebuilding the Usage dashboard from daily_snapshots\ ...
echo.
"%PY%" update_dashboard.py

echo.
echo ============================================================
echo  Done. Read the "DAILY LOAD & MATCH REPORT" above:
echo    - anything under INVESTIGATE may be a school being missed
echo    - a STALE warning means today's file isn't in the folder
echo  The same report is saved in daily_report.txt
echo ============================================================
echo.
pause
