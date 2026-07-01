@echo off
cd /d "%~dp0"
echo Starting AI Image Upscaler and Sharpener EX...
.\venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application terminated unexpectedly.
    pause
)
