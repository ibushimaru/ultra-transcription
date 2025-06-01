@echo off
REM Interactive helper for users who don't know command line
cd /d "%~dp0"

:menu
cls
echo ========================================
echo Ultra Audio Transcription v3.1.0
echo Powered by Whisper Large-v3 Turbo
echo ========================================
echo.
echo [1] Transcribe audio file
echo [2] Transcribe with options
echo [3] Check for updates
echo [4] Open documentation
echo [5] Exit
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" goto :simple_transcribe
if "%choice%"=="2" goto :advanced_transcribe
if "%choice%"=="3" goto :check_updates
if "%choice%"=="4" goto :open_docs
if "%choice%"=="5" exit /b 0
goto :menu

:simple_transcribe
cls
echo ========================================
echo Simple Transcription
echo ========================================
echo.
echo Drag and drop your audio file here, then press Enter:
echo (Or type the full path)
echo.
set /p audiofile="Audio file: "

REM Remove quotes if present
set audiofile=%audiofile:"=%

if not exist "%audiofile%" (
    echo.
    echo ERROR: File not found!
    pause
    goto :menu
)

REM Generate output name based on input file
for %%F in ("%audiofile%") do set basename=%%~nF
set outputname=%basename%_transcribed

echo.
echo Processing "%audiofile%"...
echo Output will be saved as: %outputname%
echo.

REM Run transcription
call ultra-transcribe "%audiofile%" -o "%outputname%"

echo.
echo ========================================
echo Transcription Complete!
echo ========================================
echo Output files:
echo - %outputname%_ultra_precision.txt
echo - %outputname%_ultra_precision.json
echo - %outputname%_ultra_precision.csv
echo - %outputname%_ultra_precision.srt
echo.
pause
goto :menu

:advanced_transcribe
cls
echo ========================================
echo Advanced Transcription Options
echo ========================================
echo.
set /p audiofile="Audio file (drag & drop): "
set audiofile=%audiofile:"=%

echo.
echo Options:
echo [1] Keep filler words (なるほど、ええ etc.) - Default
echo [2] Remove filler words (clean transcript)
set /p filler_choice="Select (1-2): "

echo.
echo Speaker Recognition:
echo [1] Enable speaker recognition - Default
echo [2] Disable for faster processing
set /p speaker_choice="Select (1-2): "

for %%F in ("%audiofile%") do set basename=%%~nF
set outputname=%basename%_transcribed

REM Build command
set cmd=call ultra-transcribe "%audiofile%" -o "%outputname%"
if "%filler_choice%"=="2" set cmd=%cmd% --no-fillers
if "%speaker_choice%"=="2" set cmd=%cmd% --no-speaker

echo.
echo Processing with your selected options...
%cmd%

echo.
echo Transcription Complete!
pause
goto :menu

:check_updates
cls
echo ========================================
echo Checking for Updates
echo ========================================
echo.

REM Simple version check
echo Current version: v3.1.0
echo.
echo Checking GitHub for updates...

REM Try to fetch latest release info
powershell -Command "try { $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ibushimaru/ultra-transcription/releases/latest'; Write-Host \"Latest version: $($release.tag_name)\"; if ($release.tag_name -ne 'v3.1.0') { Write-Host 'Update available!' -ForegroundColor Yellow; Write-Host \"Download from: $($release.html_url)\" } else { Write-Host 'You have the latest version!' -ForegroundColor Green } } catch { Write-Host 'Could not check for updates. Check your internet connection.' -ForegroundColor Red }"

echo.
pause
goto :menu

:open_docs
echo Opening documentation...
if exist "README.md" (
    start README.md
) else (
    start https://github.com/ibushimaru/ultra-transcription/blob/master/README.md
)
goto :menu