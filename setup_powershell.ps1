# Setup PowerShell alias for Ultra Audio Transcription
# Run this script to add convenient PowerShell commands

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$profileDir = Split-Path -Parent $PROFILE

# Create PowerShell profile directory if it doesn't exist
if (!(Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}

# Add function to PowerShell profile
$functionContent = @"

# Ultra Audio Transcription function
function ultra-transcribe {
    `$pythonPath = "$scriptDir\venv\Scripts\python.exe"
    if (Test-Path `$pythonPath) {
        & `$pythonPath -m transcription.rapid_ultra_processor `$args
    } else {
        Write-Host "Error: Virtual environment not found. Please run install.bat first." -ForegroundColor Red
    }
}

# Alias for convenience
Set-Alias -Name transcribe -Value ultra-transcribe
"@

# Check if function already exists in profile
if (Test-Path $PROFILE) {
    $profileContent = Get-Content $PROFILE -Raw
    if ($profileContent -notmatch "Ultra Audio Transcription function") {
        Add-Content -Path $PROFILE -Value $functionContent
        Write-Host "✅ Ultra Audio Transcription function added to PowerShell profile" -ForegroundColor Green
    } else {
        Write-Host "ℹ️ Function already exists in PowerShell profile" -ForegroundColor Yellow
    }
} else {
    # Create profile if it doesn't exist
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
    Add-Content -Path $PROFILE -Value $functionContent
    Write-Host "✅ PowerShell profile created and function added" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete! You can now use:" -ForegroundColor Cyan
Write-Host "  ultra-transcribe audio.mp3 -o result" -ForegroundColor White
Write-Host "  transcribe audio.mp3 -o result" -ForegroundColor White
Write-Host ""
Write-Host "Note: Restart PowerShell or run '. `$PROFILE' to load the new functions" -ForegroundColor Yellow