@echo off
set REQUIRED_PYTHON=3.9

:: Check Python version using py -3.9
for /f "tokens=2 delims= " %%a in ('py -3.9 --version 2^>^&1') do set PYTHON_VERSION=%%a
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do set PYTHON_MAJOR=%%a.%%b

if not "%PYTHON_MAJOR%"=="%REQUIRED_PYTHON%" (
    echo Python %REQUIRED_PYTHON% required, found: %PYTHON_MAJOR%
    echo Please install Python %REQUIRED_PYTHON% before continuing.
    exit /b 1
)

echo Python %PYTHON_MAJOR% detected - OK

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    py -3.9 -m venv .venv
)

:: Activate virtual environment
call .venv\Scripts\activate

:: Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Environment successfully configured with CUDA support.
pause