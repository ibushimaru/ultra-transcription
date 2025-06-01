@echo off
REM Add Ultra Audio Transcription to Windows PATH
REM This allows running 'ultra-transcribe' from anywhere

echo ========================================
echo Ultra Audio Transcription
echo Add to System PATH
echo ========================================
echo.

REM Check if already in PATH
echo %PATH% | find /i "%CD%" >nul
if not errorlevel 1 (
    echo This directory is already in your PATH.
    pause
    exit /b 0
)

REM Ask for confirmation
echo This will add the following directory to your PATH:
echo %CD%
echo.
echo This allows you to run 'ultra-transcribe' from any location.
echo.
set /p confirm="Do you want to continue? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Cancelled.
    pause
    exit /b 0
)

REM Add to user PATH
setx PATH "%PATH%;%CD%"

echo.
echo ========================================
echo Success!
echo ========================================
echo.
echo The directory has been added to your PATH.
echo You can now use 'ultra-transcribe' from any location.
echo.
echo NOTE: You need to open a new Command Prompt or PowerShell
echo      window for the changes to take effect.
echo.
pause