@echo off
REM =====================================================
REM DAILY GRADUAL COMMIT SCRIPT
REM Commits 5 files per day until project is complete
REM =====================================================

cd /d "%~dp0"

echo =====================================================
echo DAILY GITHUB COMMIT - Gradual Upload
echo =====================================================
echo.

python gradual_commit.py

echo.
pause
