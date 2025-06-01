@echo off
echo ========================================
echo Ultra Audio Transcription Setup v3.0.0
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8 or higher is required
    pause
    exit /b 1
)

echo Python version OK
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA support...
echo This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install the package
echo.
echo Installing Ultra Audio Transcription...
pip install -e .

REM Install optional GPU dependencies
echo.
echo Installing GPU acceleration components...
pip install -e .[gpu]

REM Download Whisper models
echo.
echo Downloading Whisper Large-v3-turbo model...
echo This will take some time on first run...
python -c "import whisper; whisper.load_model('large-v3-turbo')"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To use Ultra Audio Transcription:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run: ultra-transcribe your_audio.mp3 -o output
echo.
echo For Japanese transcription with filler words:
echo ultra-transcribe audio.mp3 -o output --preserve-fillers
echo.
pause