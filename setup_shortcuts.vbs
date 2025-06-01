' Create desktop and start menu shortcuts for Ultra Audio Transcription
Option Explicit

Dim objShell, objFSO, strDesktop, strStartMenu, strAppPath
Dim objShortcut, strIconPath

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get paths
strDesktop = objShell.SpecialFolders("Desktop")
strStartMenu = objShell.SpecialFolders("Programs")
strAppPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Create desktop shortcut
Set objShortcut = objShell.CreateShortcut(strDesktop & "\Ultra Audio Transcription.lnk")
objShortcut.TargetPath = strAppPath & "\START.bat"
objShortcut.WorkingDirectory = strAppPath
objShortcut.WindowStyle = 1
objShortcut.Description = "Ultra Audio Transcription - Powered by Whisper AI"

' Set icon if available
If objFSO.FileExists(strAppPath & "\app.ico") Then
    objShortcut.IconLocation = strAppPath & "\app.ico"
Else
    objShortcut.IconLocation = "%SystemRoot%\System32\SHELL32.dll,264"
End If

objShortcut.Save

' Create Start Menu folder and shortcut
Dim strStartMenuFolder
strStartMenuFolder = strStartMenu & "\Ultra Audio Transcription"

If Not objFSO.FolderExists(strStartMenuFolder) Then
    objFSO.CreateFolder(strStartMenuFolder)
End If

' Main shortcut
Set objShortcut = objShell.CreateShortcut(strStartMenuFolder & "\Ultra Audio Transcription.lnk")
objShortcut.TargetPath = strAppPath & "\START.bat"
objShortcut.WorkingDirectory = strAppPath
objShortcut.WindowStyle = 1
objShortcut.Description = "Ultra Audio Transcription - Powered by Whisper AI"

If objFSO.FileExists(strAppPath & "\app.ico") Then
    objShortcut.IconLocation = strAppPath & "\app.ico"
Else
    objShortcut.IconLocation = "%SystemRoot%\System32\SHELL32.dll,264"
End If

objShortcut.Save

' Documentation shortcut
If objFSO.FileExists(strAppPath & "\README.md") Then
    Set objShortcut = objShell.CreateShortcut(strStartMenuFolder & "\Documentation.lnk")
    objShortcut.TargetPath = strAppPath & "\README.md"
    objShortcut.WorkingDirectory = strAppPath
    objShortcut.Description = "Ultra Audio Transcription Documentation"
    objShortcut.Save
End If

' Uninstall shortcut
Set objShortcut = objShell.CreateShortcut(strStartMenuFolder & "\Uninstall.lnk")
objShortcut.TargetPath = strAppPath & "\uninstall.bat"
objShortcut.WorkingDirectory = strAppPath
objShortcut.Description = "Uninstall Ultra Audio Transcription"
objShortcut.IconLocation = "%SystemRoot%\System32\SHELL32.dll,131"
objShortcut.Save

MsgBox "Shortcuts created successfully!" & vbCrLf & vbCrLf & _
       "Desktop: Ultra Audio Transcription" & vbCrLf & _
       "Start Menu: Ultra Audio Transcription folder", _
       vbInformation, "Setup Complete"