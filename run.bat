@echo off
echo ASL Translator and Emotion Communicator
echo =====================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.x from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Check if requirements are installed
echo Checking dependencies...
pip show opencv-python >nul 2>&1
if %errorlevel% neq 0 (
    echo Some dependencies are not installed.
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install dependencies.
        echo Please try running: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

:: Run the application
echo Starting ASL Translator and Emotion Communicator...
echo.
python run.py %*

:: Check if application exited with an error
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error (code %errorlevel%).
    echo Please check the logs for more information.
    echo.
    pause
    exit /b %errorlevel%
)

exit /b 0