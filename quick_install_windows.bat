@echo off
REM Quick installation script for Windows users
REM This script installs dependencies directly without pyproject.toml

echo ========================================
echo Ultra Audio Transcription v3.0.0
echo Quick Windows Installation
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install dependencies
echo.
echo Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found, installing packages directly...
    pip install openai-whisper>=20231117
    pip install pyannote.audio>=3.1.0
    pip install librosa>=0.10.0
    pip install soundfile>=0.12.0
    pip install pydub>=0.25.0
    pip install noisereduce>=3.0.0
    pip install ffmpeg-python>=0.2.0
    pip install pandas>=1.5.0
    pip install scipy>=1.9.0
    pip install scikit-learn>=1.1.0
    pip install click>=8.0.0
    pip install tqdm>=4.64.0
    pip install rich>=13.0.0
    pip install colorama>=0.4.6
    pip install pydantic>=2.0.0
    pip install jsonschema>=4.17.0
)

REM Download Whisper models
echo.
echo Downloading Whisper Large-v3-turbo model...
echo This will take some time on first run...
python -c "import whisper; whisper.load_model('large-v3-turbo')" 2>nul || (
    echo Note: Model will be downloaded on first use
)

REM Create convenience scripts
echo.
echo Creating convenience scripts...

REM Create ultra-transcribe command
echo @echo off > venv\Scripts\ultra-transcribe.bat
echo python -m transcription.rapid_ultra_processor %%* >> venv\Scripts\ultra-transcribe.bat

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To use Ultra Audio Transcription:
echo 1. Run: ultra-transcribe your_audio.mp3 -o output
echo.
echo For more options:
echo ultra-transcribe --help
echo.
pause