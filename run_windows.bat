@echo off
REM Direct runner for Ultra Audio Transcription on Windows
REM Use this if you have issues with the installed command

if not defined VIRTUAL_ENV (
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    )
)

if "%~1"=="" (
    echo Ultra Audio Transcription v3.0.1 - Direct Runner
    echo.
    echo Usage: run_windows audio_file [options]
    echo.
    echo Options:
    echo   -o OUTPUT         Output file base name (required)
    echo   --no-fillers      Disable filler word preservation
    echo   --no-speaker      Disable speaker recognition
    echo   --model MODEL     Whisper model (default: large-v3-turbo)
    echo   --language LANG   Language code (default: ja)
    echo.
    echo Examples:
    echo   run_windows interview.mp3 -o interview_result
    echo   run_windows meeting.wav -o meeting --no-speaker
    echo.
    exit /b 1
)

REM Check if Python modules are accessible
python -c "import transcription" 2>nul
if errorlevel 1 (
    echo ERROR: transcription module not found
    echo Please ensure you're in the correct directory and have installed the package
    exit /b 1
)

REM Run the transcription
echo Running Ultra Audio Transcription...
python -m transcription.rapid_ultra_processor %*