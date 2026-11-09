@echo off
rem SilentSpeech: open the live app straight away (after start.bat has set everything up once).
rem Usage: app.bat            (person s01)
rem        app.bat --speaker s02
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run start.bat first - it installs everything and calibrates.
    pause
    exit /b 1
)
venv\Scripts\python app.py %*
if errorlevel 1 pause
