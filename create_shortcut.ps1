# Create desktop shortcut for Ultra Audio Transcription
# Run this in PowerShell with: .\create_shortcut.ps1

$WshShell = New-Object -comObject WScript.Shell
$desktop = [System.Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop "Ultra Transcribe.lnk"

# Create shortcut
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$PSScriptRoot\ultra-transcribe.ps1`""
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.IconLocation = "shell32.dll,264"
$Shortcut.Description = "Ultra Audio Transcription - Whisper Turbo"
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "You can now drag and drop audio files onto the shortcut." -ForegroundColor Cyan