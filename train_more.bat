@echo off
rem SilentSpeech: record more examples of every phrase so recognition gets more accurate.
rem Usage: train_more.bat            (person s01)
rem        train_more.bat --speaker s02
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run start.bat first - it installs everything and calibrates.
    pause
    exit /b 1
)
venv\Scripts\python start.py --train-more --speaker s01 %*
pause
