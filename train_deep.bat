@echo off
rem SilentSpeech: train the deep model (a neural-network ensemble) on your own clips.
rem Usage: train_deep.bat            (person s01)
rem        train_deep.bat --speaker s02
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run start.bat first - it installs everything and calibrates.
    pause
    exit /b 1
)
venv\Scripts\python train_deep.py --speaker s01 %*
pause
