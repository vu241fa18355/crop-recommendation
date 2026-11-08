@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found. Follow START_HERE.md first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
