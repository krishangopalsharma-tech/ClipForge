@echo off
REM ============================================================
REM  ClipForge - one-click launcher
REM  Starts the local server on http://127.0.0.1:8000 and opens
REM  your browser once it's ready. Close this window to stop.
REM ============================================================
setlocal
cd /d "%~dp0"

set "HF_HOME=%~dp0models"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PATH=%~dp0ffmpeg;%~dp0python;%PATH%"

echo.
echo   Starting ClipForge...
echo   First launch downloads the AI model (~1.5 GB) - please wait.
echo   Browser opens automatically when ready. Keep this window open.
echo.

REM Open browser once /health returns, then keep server in foreground.
start "" powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "for($i=0;$i -lt 900;$i++){try{if((Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200){Start-Process 'http://127.0.0.1:8000';break}}catch{};Start-Sleep -Seconds 2}"

"%~dp0python\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo   ClipForge stopped. You can close this window.
pause >nul
