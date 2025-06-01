# Auto-update script for Ultra Audio Transcription
param(
    [switch]$Force
)

$currentVersion = "v3.1.0"
$repoUrl = "https://api.github.com/repos/ibushimaru/ultra-transcription/releases/latest"
$installPath = $PSScriptRoot

Write-Host "Ultra Audio Transcription - Auto Update" -ForegroundColor Cyan
Write-Host "Current version: $currentVersion" -ForegroundColor Gray

try {
    # Get latest release info
    $release = Invoke-RestMethod -Uri $repoUrl
    $latestVersion = $release.tag_name
    
    Write-Host "Latest version: $latestVersion" -ForegroundColor Gray
    
    if ($latestVersion -eq $currentVersion -and !$Force) {
        Write-Host "You already have the latest version!" -ForegroundColor Green
        return
    }
    
    # Find the zip asset
    $zipAsset = $release.assets | Where-Object { $_.name -like "*.zip" } | Select-Object -First 1
    
    if (!$zipAsset) {
        Write-Host "No downloadable package found for the latest release." -ForegroundColor Yellow
        Write-Host "Please download manually from: $($release.html_url)" -ForegroundColor Yellow
        return
    }
    
    Write-Host "`nUpdate available: $currentVersion -> $latestVersion" -ForegroundColor Yellow
    $confirm = Read-Host "Do you want to update? (Y/N)"
    
    if ($confirm -ne 'Y') {
        Write-Host "Update cancelled." -ForegroundColor Gray
        return
    }
    
    # Download update
    $tempPath = Join-Path $env:TEMP "ultra-transcription-update.zip"
    Write-Host "Downloading update..." -ForegroundColor Cyan
    
    Invoke-WebRequest -Uri $zipAsset.browser_download_url -OutFile $tempPath
    
    # Backup current installation
    $backupPath = "$installPath.backup"
    Write-Host "Creating backup..." -ForegroundColor Cyan
    
    if (Test-Path $backupPath) {
        Remove-Item $backupPath -Recurse -Force
    }
    
    # Exclude large folders from backup
    $itemsToBackup = Get-ChildItem $installPath -Exclude "venv", ".git", "__pycache__", "*.mp3", "*.wav"
    $itemsToBackup | Copy-Item -Destination $backupPath -Recurse -Force
    
    # Extract update
    Write-Host "Installing update..." -ForegroundColor Cyan
    Expand-Archive -Path $tempPath -DestinationPath $env:TEMP -Force
    
    # Find extracted folder
    $extractedFolder = Get-ChildItem $env:TEMP | Where-Object { $_.Name -like "*ultra-transcription*" -and $_.PSIsContainer } | Select-Object -First 1
    
    if ($extractedFolder) {
        # Copy new files
        Copy-Item -Path "$($extractedFolder.FullName)\*" -Destination $installPath -Recurse -Force
        
        # Clean up
        Remove-Item $tempPath -Force
        Remove-Item $extractedFolder.FullName -Recurse -Force
        
        Write-Host "`nUpdate completed successfully!" -ForegroundColor Green
        Write-Host "Please restart the application." -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: Could not find extracted files." -ForegroundColor Red
        Write-Host "Please update manually from: $($release.html_url)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "`nERROR: Update failed - $_" -ForegroundColor Red
    Write-Host "Please download manually from: https://github.com/ibushimaru/ultra-transcription/releases" -ForegroundColor Yellow
}