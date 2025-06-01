@echo off
REM Test script for --no-speaker option

echo ========================================
echo Testing --no-speaker Option
echo ========================================
echo.

if not exist venv\Scripts\python.exe (
    echo ERROR: Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

REM Create test directory
if not exist test_outputs mkdir test_outputs

echo Test 1: WITH speaker recognition (default)
echo ----------------------------------------
venv\Scripts\python.exe -m transcription.rapid_ultra_processor ^
    testdata\test_90s.mp3 -o test_outputs\test_with_speaker ^
    --chunk-size 1.0

echo.
echo.
echo Test 2: WITHOUT speaker recognition (--no-speaker)
echo ----------------------------------------
venv\Scripts\python.exe -m transcription.rapid_ultra_processor ^
    testdata\test_90s.mp3 -o test_outputs\test_no_speaker ^
    --chunk-size 1.0 --no-speaker

echo.
echo ========================================
echo Tests Complete
echo ========================================
echo.
echo Check the console output above for:
echo 1. "Processing settings: speaker_recognition=True" (first test)
echo 2. "Processing settings: speaker_recognition=False" (second test)
echo 3. "Speaker recognition enabled" vs "Speaker recognition disabled" messages
echo.
echo Compare processing times - --no-speaker should be faster
echo.
pause