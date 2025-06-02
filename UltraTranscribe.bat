@echo off
cd /d "%~dp0"
python UltraTranscribe.py %*
if errorlevel 1 pause