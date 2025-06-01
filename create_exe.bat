@echo off
REM Create standalone exe for GUI

echo Creating standalone executable...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install PyInstaller if not present
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Create exe
echo Building transcribe-gui.exe...
pyinstaller --onefile --windowed --name transcribe-gui ^
    --icon=app.ico ^
    --add-data "transcription;transcription" ^
    --hidden-import=whisper ^
    --hidden-import=librosa ^
    --hidden-import=pyannote ^
    transcribe_gui.py

if exist dist\transcribe-gui.exe (
    echo.
    echo Success! Created: dist\transcribe-gui.exe
    copy dist\transcribe-gui.exe . >nul
    echo Copied to: transcribe-gui.exe
) else (
    echo.
    echo Failed to create exe. GUI will run from Python.
)

pause