@echo off
rem SilentSpeech: double-click to set up (first time) and start.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo First run: creating a Python environment and installing libraries.
    echo This downloads about 1 GB and can take 5-15 minutes.
    py -3.12 -m venv venv 2>nul || py -3.11 -m venv venv 2>nul || py -3.10 -m venv venv 2>nul || python -m venv venv
    if not exist venv\Scripts\python.exe (
        echo Could not create the environment. Install Python 3.10, 3.11 or 3.12 from python.org and try again.
        pause
        exit /b 1
    )
    venv\Scripts\python -m pip install --upgrade pip
    venv\Scripts\python -m pip install -r requirements.txt || (
        echo Installing the libraries failed - see the messages above.
        pause
        exit /b 1
    )
)
venv\Scripts\python start.py %*
pause
