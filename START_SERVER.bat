@echo off
echo ====================================
echo Starting HSEF Mock Server
echo ====================================
echo.
echo Server will run at: http://localhost:5000
echo.
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0"
python app\mock_app.py

pause
