@echo off
REM Ultra Audio Transcription - Windows Batch File
REM Convenience wrapper for transcription

if "%~1"=="" (
    echo Ultra Audio Transcription v3.0.0 - Turbo Edition
    echo.
    echo Usage: ultra-transcribe audio_file [options]
    echo.
    echo Options:
    echo   -o OUTPUT         Output file base name (required)
    echo   --no-fillers      Disable filler word preservation
    echo   --no-speaker      Disable speaker recognition
    echo   --model MODEL     Whisper model (default: large-v3-turbo)
    echo   --language LANG   Language code (default: ja)
    echo.
    echo Examples:
    echo   ultra-transcribe interview.mp3 -o interview_result
    echo   ultra-transcribe meeting.wav -o meeting --no-speaker
    echo.
    exit /b 1
)

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    if exist venv\Scripts\activate.bat (
        echo Activating virtual environment...
        call venv\Scripts\activate.bat
    ) else (
        echo ERROR: Virtual environment not found. Please run setup_windows.bat first.
        exit /b 1
    )
)

REM Default to preserve fillers for Japanese
set DEFAULT_ARGS=--preserve-fillers

REM Check if user wants no fillers
echo %* | find "--no-fillers" >nul
if not errorlevel 1 (
    set DEFAULT_ARGS=
    set ARGS=%*
    set ARGS=!ARGS:--no-fillers=!
) else (
    set ARGS=%* %DEFAULT_ARGS%
)

REM Run the transcription
echo Running Ultra Audio Transcription...
python -m transcription.rapid_ultra_processor %ARGS%

if errorlevel 1 (
    echo.
    echo ERROR: Transcription failed. Please check the error messages above.
    exit /b 1
)

echo.
echo Transcription complete!