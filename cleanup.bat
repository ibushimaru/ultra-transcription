@echo off
REM Cleanup script to remove accidentally created files

echo ========================================
echo Cleaning up installation artifacts
echo ========================================
echo.

REM Remove files that look like version numbers
for %%f in (0.*.* 1.*.* 2.*.* 3.*.* 4.*.* 5.*.* 6.*.* 7.*.* 8.*.* 9.*.*) do (
    if exist "%%f" (
        echo Removing: %%f
        del "%%f" 2>nul
    )
)

REM Remove common package version patterns
for %%f in (*=*.* *^>=*.* *^<=*.* *~=*.*) do (
    if exist "%%f" (
        echo Removing: %%f
        del "%%f" 2>nul
    )
)

echo.
echo Cleanup complete!
echo.
pause