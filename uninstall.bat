@echo off
REM Uninstaller for Ultra Audio Transcription

echo ========================================
echo Ultra Audio Transcription - Uninstaller
echo ========================================
echo.
echo This will remove:
echo  - Virtual environment
echo  - Downloaded models
echo  - Desktop shortcuts
echo  - Start menu entries
echo.
echo Your audio files and transcriptions will NOT be deleted.
echo.
set /p confirm="Are you sure you want to uninstall? (Y/N): "

if /i not "%confirm%"=="Y" (
    echo Uninstall cancelled.
    pause
    exit /b 0
)

cd /d "%~dp0"

echo.
echo Removing virtual environment...
if exist venv rmdir /s /q venv

echo Removing cache and models...
if exist .cache rmdir /s /q .cache
if exist models rmdir /s /q models
if exist __pycache__ rmdir /s /q __pycache__
if exist transcription\__pycache__ rmdir /s /q transcription\__pycache__

echo Removing setup markers...
if exist .setup_complete del .setup_complete

echo Removing shortcuts...
set "desktop=%USERPROFILE%\Desktop"
set "startmenu=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

if exist "%desktop%\Ultra Audio Transcription.lnk" del "%desktop%\Ultra Audio Transcription.lnk"
if exist "%startmenu%\Ultra Audio Transcription" rmdir /s /q "%startmenu%\Ultra Audio Transcription"

echo.
echo ========================================
echo Uninstall Complete
echo ========================================
echo.
echo The following have been removed:
echo  - Python virtual environment
echo  - Downloaded AI models
echo  - Shortcuts
echo.
echo The following have been kept:
echo  - Program files (can be deleted manually)
echo  - Your transcription outputs
echo.
pause