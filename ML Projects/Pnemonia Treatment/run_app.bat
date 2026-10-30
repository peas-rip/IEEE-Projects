@echo off
REM Pneumonia Detection — one-command launcher (Windows)
REM Creates venv if missing, installs deps, starts the app, opens the browser.
setlocal

cd /d "%~dp0"

set VENV_DIR=.venv
set PYTHON=python

if not exist "%VENV_DIR%" (
  echo Creating virtual environment...
  %PYTHON% -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt
pip install -q -e ".[serve]"

REM Build frontend if dist/ is missing
if not exist "frontend\dist" (
  where npm >nul 2>nul
  if %errorlevel% equ 0 (
    echo Building frontend...
    pushd frontend
    npm install --silent
    npm run build --silent
    popd
  ) else (
    echo WARNING: frontend\dist missing and npm not found.
    echo Install Node.js or pre-build the frontend before running this script.
    echo The API is still available at http://localhost:8000/api/docs
  )
)

echo.
echo Starting Pneumonia Detection on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

start "" http://localhost:8000
python app.main

endlocal
