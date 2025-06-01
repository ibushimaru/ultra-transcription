@echo off
REM ========================================
REM Ultra Audio Transcription - One-Click Start
REM ========================================
REM This is the ONLY file users need to run
REM Everything else is automated

cd /d "%~dp0"

REM Check if already set up
if exist ".setup_complete" (
    goto :run_app
)

REM First time setup
echo ========================================
echo Ultra Audio Transcription
echo First Time Setup (This only happens once)
echo ========================================
echo.
echo This will take 5-10 minutes...
echo.

REM Run installation
call install.bat
if errorlevel 1 (
    echo Setup failed. Please check error messages above.
    pause
    exit /b 1
)

REM Mark setup as complete
echo Setup completed on %date% %time% > .setup_complete

:run_app
REM Launch the GUI
if exist "transcribe-gui.exe" (
    start transcribe-gui.exe
) else (
    REM Fallback to command line helper
    call transcribe-helper.bat
)