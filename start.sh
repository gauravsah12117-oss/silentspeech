#!/usr/bin/env sh
# SilentSpeech on macOS / Linux: set up (first time) and start.  Usage: sh start.sh
set -e
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "First run: creating a Python environment and installing libraries (about 1 GB)."
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v "$py" >/dev/null 2>&1; then "$py" -m venv venv && break; fi
    done
    venv/bin/python -m pip install --upgrade pip
    venv/bin/python -m pip install -r requirements.txt
fi
exec venv/bin/python start.py "$@"
