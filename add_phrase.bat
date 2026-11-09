@echo off
rem SilentSpeech: add your own phrase (any words, any language) and teach it.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run start.bat first - it installs everything and calibrates.
    pause
    exit /b 1
)
venv\Scripts\python add_phrase.py %*
pause
