@echo off
REM Diagnostic and repair tool for Ultra Audio Transcription

echo ========================================
echo Ultra Audio Transcription - Diagnostics
echo ========================================
echo.
echo Checking system status...
echo.

set issues=0

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo    ❌ Python not found or not in PATH
    echo       Solution: Install Python 3.8+ from https://python.org
    set /a issues+=1
) else (
    python --version
    echo    ✅ Python is installed
)

REM Check virtual environment
echo.
echo [2/6] Checking virtual environment...
if exist venv\Scripts\python.exe (
    echo    ✅ Virtual environment exists
) else (
    echo    ❌ Virtual environment not found
    echo       Solution: Run install.bat
    set /a issues+=1
)

REM Check key packages
echo.
echo [3/6] Checking required packages...
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe -c "import whisper" >nul 2>&1
    if errorlevel 1 (
        echo    ❌ Whisper package not installed
        set /a issues+=1
    ) else (
        echo    ✅ Whisper package installed
    )
    
    venv\Scripts\python.exe -c "import librosa" >nul 2>&1
    if errorlevel 1 (
        echo    ❌ Librosa package not installed
        set /a issues+=1
    ) else (
        echo    ✅ Librosa package installed
    )
)

REM Check model files
echo.
echo [4/6] Checking AI models...
set model_found=0
if exist "%USERPROFILE%\.cache\whisper\large-v3-turbo.pt" set model_found=1
if exist ".cache\whisper\large-v3-turbo.pt" set model_found=1

if %model_found%==1 (
    echo    ✅ Turbo model is downloaded
) else (
    echo    ⚠️  Turbo model not downloaded (will download on first use)
)

REM Check PATH setup
echo.
echo [5/6] Checking PATH configuration...
echo %PATH% | find /i "%CD%" >nul
if errorlevel 1 (
    echo    ⚠️  Current directory not in PATH
    echo       Solution: Run add_to_path.bat for global access
) else (
    echo    ✅ PATH is configured
)

REM Check file permissions
echo.
echo [6/6] Checking file permissions...
echo test > test_write.tmp 2>nul
if exist test_write.tmp (
    del test_write.tmp
    echo    ✅ Write permissions OK
) else (
    echo    ❌ No write permissions in current directory
    set /a issues+=1
)

REM Summary
echo.
echo ========================================
echo Diagnostic Summary
echo ========================================
echo.

if %issues%==0 (
    echo ✅ All checks passed! System is ready.
    echo.
    echo You can now:
    echo  - Run START.bat to begin
    echo  - Use ultra-transcribe command
    echo  - Drag audio files to shortcuts
) else (
    echo ❌ Found %issues% issue(s) that need attention.
    echo.
    echo Recommended actions:
    
    if not exist venv\Scripts\python.exe (
        echo  1. Run install.bat to set up the environment
    )
    
    echo.
    set /p repair="Would you like to attempt automatic repair? (Y/N): "
    if /i "!repair!"=="Y" (
        echo.
        echo Starting automatic repair...
        call install.bat
    )
)

echo.
pause