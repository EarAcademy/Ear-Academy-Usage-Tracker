@echo off
REM ============================================================
REM  Ear Academy — Upload today's daily usage snapshot
REM
REM  TEMPORARY SETUP: while Rus is running ALL THREE dashboards
REM  from his side, this is the ONLY thing to run from here.
REM
REM  What this does:
REM    1. Pulls the latest from GitHub
REM    2. Commits + pushes whatever new/changed files are in
REM       daily_snapshots\ (the .xlsx you just dropped in there)
REM
REM  What this does NOT do:
REM    - Does NOT run update_dashboard.py
REM    - Does NOT touch or rebuild index.html
REM    - Does NOT rebuild or push the live usage dashboard
REM  That part is Rus's job now — his update_all_dashboards.sh
REM  will pick up the new snapshot file next time he runs it and
REM  rebuild everything (usage + sales + velocity) from his side.
REM
REM  HOW TO USE: save today's snapshot .xlsx into the
REM  daily_snapshots\ folder (same folder this .bat lives next
REM  to), then double-click this file.
REM ============================================================

cd /d "%~dp0"

echo.
echo [1/2] Pulling latest from GitHub...
echo ----------------------------------------------------------
git pull origin main
if not %ERRORLEVEL%==0 (
  echo.
  echo   ^!^! git pull failed — resolve this before continuing.
  echo   ^   Nothing was uploaded.
  echo.
  pause
  exit /b 1
)

echo.
echo [2/2] Uploading today's snapshot(s)...
echo ----------------------------------------------------------
git add daily_snapshots\
git commit -m "Add daily usage snapshot %DATE%"
if %ERRORLEVEL%==0 (
  git push origin main
  echo.
  echo   -^> Uploaded. Rus's next full dashboard run will pick this up
  echo      and rebuild the usage dashboard from it.
) else (
  echo.
  echo   -^> Nothing new to upload — no new snapshot file found in
  echo      daily_snapshots\, or it's already been uploaded.
)

echo.
echo ============================================================
echo  Done. This did NOT rebuild or push the live dashboard —
echo  Rus's update_all_dashboards.sh handles that from here.
echo ============================================================
echo.
pause
