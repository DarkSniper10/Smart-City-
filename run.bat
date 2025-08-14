@echo off
REM Change to your project directory
cd /d "C:\Users\anant\OneDrive\Desktop\smart"

REM Activate the virtual environment
call npr_env\Scripts\activate.bat

REM Run the Python script
python test_ocr.py

REM Pause so you can see any final output
pause
