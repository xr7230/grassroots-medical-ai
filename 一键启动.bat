@echo off
chcp 65001 >nul
title AI Medical Record System

echo ========================================
echo    AI Medical Record System - Startup
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
set BACKEND_DIR=%SCRIPT_DIR%backend
set FRONTEND_FILE=%SCRIPT_DIR%frontend\index.html

echo [1/4] Checking directories...
echo   Backend: %BACKEND_DIR%
echo   Frontend: %FRONTEND_FILE%

if not exist "%BACKEND_DIR%" (
    echo [ERROR] Backend directory not found
    pause
    exit /b 1
)

if not exist "%FRONTEND_FILE%" (
    echo [ERROR] Frontend file not found
    pause
    exit /b 1
)

echo.
echo [2/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)
echo   Python OK

echo.
echo [3/4] Starting backend service...
cd /d "%BACKEND_DIR%"
echo   Working dir: %CD%
start "Backend Service" cmd /k python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

echo.
echo [4/4] Waiting for service...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo    Startup Complete!
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Opening frontend...
echo.

start "" "%FRONTEND_FILE%"

echo Press any key to close...
pause >nul
