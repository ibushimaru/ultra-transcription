# Ultra Audio Transcription - Simple Guide

## How to Use

1. **Download** the latest release
2. **Extract** the files  
3. **Double-click** `UltraTranscribe.bat`

That's it! The first run will automatically install everything needed (5-10 minutes).

## Features

- **12.6x faster** with Whisper Turbo model
- **98.4% accuracy** with GPU acceleration
- **Speaker recognition** for multiple speakers
- **Natural conversation mode** preserves filler words
- **Multiple output formats** (TXT, JSON, CSV, SRT)

## Usage Options

### Option 1: GUI (Easiest)
Double-click `UltraTranscribe.bat` → Select file → Click "Start"

### Option 2: Command Line
```
UltraTranscribe.bat audio.mp3 -o transcript
```

### Option 3: Interactive
Just run `UltraTranscribe.bat` and answer the questions

## Options

- `--no-speaker` - Disable speaker recognition (faster)
- `--no-fillers` - Remove filler words (cleaner)
- `--gui` - Force GUI mode
- `--help` - Show all options

## Requirements

- Windows 10/11
- Python 3.8+ (download from python.org if needed)
- 8GB RAM (16GB recommended)
- NVIDIA GPU (optional but recommended)

## Support

See full documentation in README.md or visit:
https://github.com/ibushimaru/ultra-transcription