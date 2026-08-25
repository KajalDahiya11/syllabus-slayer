@echo off
echo ============================================
echo  Syllabus Slayer AI Backend — Startup Script
echo ============================================
echo.

:: Navigate to backend folder
cd /d "%~dp0backend"

:: Check if virtual environment exists; create if not
if not exist ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/3] Virtual environment found.
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install/upgrade dependencies
echo [2/3] Installing dependencies...
pip install -r requirements.txt --quiet

:: Load .env and start server
echo [3/3] Starting FastAPI server on http://localhost:8000
echo.
echo  Open frontend\Syllabus_Slayer.html in your browser once the server starts.
echo  Press Ctrl+C to stop the server.
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
