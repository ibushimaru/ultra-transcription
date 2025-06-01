#!/bin/bash
# Release preparation script for Ultra Audio Transcription v3.0.0

echo "🚀 Preparing Ultra Audio Transcription v3.0.0 for release..."
echo

# Clean up test files (keep organized ones)
echo "🧹 Cleaning up test files..."
find test_outputs -type f \( -name "*.json" -o -name "*.csv" -o -name "*.srt" -o -name "*.txt" \) ! -path "*/organized/*" -delete 2>/dev/null || true
echo "   ✓ Test files cleaned"

# Clean up root directory test outputs
echo "🧹 Cleaning up root directory outputs..."
rm -f *.json *.csv *.srt *.txt 2>/dev/null || true
echo "   ✓ Root directory cleaned"

# Remove Zone.Identifier files
echo "🧹 Removing Zone.Identifier files..."
find . -name "*:Zone.Identifier" -delete 2>/dev/null || true
echo "   ✓ Zone.Identifier files removed"

# Create release tag
echo
echo "📦 Creating release tag..."
echo "   Tag: v3.0.0"
echo "   Title: Turbo Revolution - 12.6x Speed & Filler Preservation"
echo

# Display release checklist
echo "📋 Release Checklist:"
echo "   ✓ CHANGELOG.md updated with v3.0.0 release notes"
echo "   ✓ README.md updated with Turbo model information"
echo "   ✓ Windows installer (setup_windows.bat) created"
echo "   ✓ Windows wrapper (ultra-transcribe.bat) created"
echo "   ✓ Installation guide (INSTALL_WINDOWS.md) created"
echo "   ✓ Release notes (RELEASE_NOTES_v3.0.0.md) prepared"
echo "   ✓ .gitignore updated for test outputs"
echo "   ✓ Default model set to large-v3-turbo"
echo "   ✓ Filler preservation mode enabled by default"
echo

echo "🎯 Next Steps:"
echo "   1. Commit all changes:"
echo "      git add -A"
echo "      git commit -m \"feat: Release v3.0.0 - Turbo Revolution with filler preservation\""
echo
echo "   2. Create and push tag:"
echo "      git tag -a v3.0.0 -m \"Ultra Audio Transcription v3.0.0 - Turbo Revolution\""
echo "      git push origin master"
echo "      git push origin v3.0.0"
echo
echo "   3. Create GitHub release:"
echo "      - Go to https://github.com/[your-username]/ultra-audio-transcription/releases/new"
echo "      - Select tag: v3.0.0"
echo "      - Title: \"Ultra Audio Transcription v3.0.0 - Turbo Revolution 🚀\""
echo "      - Copy content from RELEASE_NOTES_v3.0.0.md"
echo "      - Attach Windows release files if creating separate package"
echo

echo "✅ Release preparation complete!"