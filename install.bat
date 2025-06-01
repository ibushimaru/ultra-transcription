@echo off
REM Ultra Audio Transcription v3.0.3 - Universal Windows Installer
REM This script combines all installation methods with automatic fallback

echo ========================================
echo Ultra Audio Transcription v3.0.3
echo Windows Installation
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA
echo.
echo Installing PyTorch with CUDA support...
echo This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Try main installation method
echo.
echo Installing Ultra Audio Transcription...

REM Method 1: Try with pyproject.toml
pip install -e . 2>nul
if not errorlevel 1 (
    goto :install_success
)

echo Standard installation failed, trying alternative method...

REM Method 2: Install requirements directly
echo Installing dependencies directly...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo Installing core packages...
    pip install "openai-whisper>=20231117"
    pip install "pyannote.audio>=3.1.0"
    pip install "librosa>=0.10.0"
    pip install "soundfile>=0.12.0"
    pip install "pydub>=0.25.0"
    pip install "noisereduce>=3.0.0"
    pip install "ffmpeg-python>=0.2.0"
    pip install "pandas>=1.5.0"
    pip install "scipy>=1.9.0"
    pip install "scikit-learn>=1.1.0"
    pip install "click>=8.0.0"
    pip install "tqdm>=4.64.0"
    pip install "rich>=13.0.0"
    pip install "colorama>=0.4.6"
    pip install "pydantic>=2.0.0"
    pip install "jsonschema>=4.17.0"
)

:install_success

REM Download Whisper model
echo.
echo Downloading Whisper Large-v3-turbo model...
echo This will take some time on first run (about 2GB)...
python -c "import whisper; whisper.load_model('large-v3-turbo')" 2>nul
if errorlevel 1 (
    echo Note: Model will be downloaded automatically on first use
)

REM Create execution scripts
echo.
echo Creating execution scripts...

REM Create ultra-transcribe.bat in project root
echo @echo off > ultra-transcribe.bat
echo REM Ultra Audio Transcription - Main execution script >> ultra-transcribe.bat
echo. >> ultra-transcribe.bat
echo if not defined VIRTUAL_ENV ( >> ultra-transcribe.bat
echo     if exist "%~dp0venv\Scripts\activate.bat" ( >> ultra-transcribe.bat
echo         call "%~dp0venv\Scripts\activate.bat" >> ultra-transcribe.bat
echo     ) >> ultra-transcribe.bat
echo ) >> ultra-transcribe.bat
echo. >> ultra-transcribe.bat
echo python -m transcription.rapid_ultra_processor %%* >> ultra-transcribe.bat

REM Create PowerShell script
echo # Ultra Audio Transcription - PowerShell execution script > ultra-transcribe.ps1
echo $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path >> ultra-transcribe.ps1
echo $venvPath = Join-Path $scriptPath "venv\Scripts\python.exe" >> ultra-transcribe.ps1
echo if (Test-Path $venvPath) { >> ultra-transcribe.ps1
echo     ^& $venvPath -m transcription.rapid_ultra_processor $args >> ultra-transcribe.ps1
echo } else { >> ultra-transcribe.ps1
echo     Write-Host "Error: Virtual environment not found. Please run install.bat first." -ForegroundColor Red >> ultra-transcribe.ps1
echo     exit 1 >> ultra-transcribe.ps1
echo } >> ultra-transcribe.ps1

REM Also create in Scripts folder if exists
if exist venv\Scripts (
    copy ultra-transcribe.bat venv\Scripts\ >nul 2>&1
    copy ultra-transcribe.ps1 venv\Scripts\ >nul 2>&1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To use Ultra Audio Transcription:
echo.
echo 1. For Japanese transcription with filler words (default):
echo    ultra-transcribe interview.mp3 -o result
echo.
echo 2. For clean transcription without fillers:
echo    ultra-transcribe interview.mp3 -o result --no-fillers
echo.
echo 3. For fast processing without speaker recognition:
echo    ultra-transcribe audio.mp3 -o result --no-speaker
echo.
echo For more options:
echo    ultra-transcribe --help
echo.
echo Output files will be created as:
echo    - result_ultra_precision.json (detailed)
echo    - result_ultra_precision.csv  (spreadsheet)
echo    - result_ultra_precision.srt  (subtitles)
echo.
pause