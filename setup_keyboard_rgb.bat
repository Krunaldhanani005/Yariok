@echo off
:: Run this ONCE as Administrator to enable Yaariok keyboard RGB control
:: After setup, smart_system.py can control keyboard RGB without admin prompts

echo ============================================================
echo   Yaariok Keyboard RGB Setup (Run as Administrator)
echo ============================================================
echo.

:: Check admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please right-click this file and choose "Run as administrator"
    pause
    exit /b 1
)

:: Find Python venv
set PYTHON=C:\Yaariok\.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    echo ERROR: Python venv not found at %PYTHON%
    pause
    exit /b 1
)

echo [1/2] Creating scheduled task for keyboard ON...
schtasks /create /tn "Yaariok_KB_On" ^
  /tr "\"%PYTHON%\" C:\Yaariok\kb_control.py on" ^
  /sc once /st 00:00 /ru SYSTEM ^
  /rl HIGHEST /f >nul 2>&1
if %errorLevel% equ 0 (echo     OK: Yaariok_KB_On) else (echo     WARNING: Task creation had issues)

echo [2/2] Creating scheduled task for keyboard OFF...
schtasks /create /tn "Yaariok_KB_Off" ^
  /tr "\"%PYTHON%\" C:\Yaariok\kb_control.py off" ^
  /sc once /st 00:00 /ru SYSTEM ^
  /rl HIGHEST /f >nul 2>&1
if %errorLevel% equ 0 (echo     OK: Yaariok_KB_Off) else (echo     WARNING: Task creation had issues)

echo.
echo Setup complete! Now say "lights on" or "lights off" to Yaariok.
echo.
pause
