@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo   ISL Sign2Speech - One-Click Launcher
echo ========================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python.exe not found on PATH.
    echo         Install Python 3.10, 3.11 or 3.12 from https://www.python.org/downloads/
    echo         (3.13+ is NOT supported because TensorFlow wheels are unavailable.)
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')"') do set PYVER=%%i
echo [INFO] Detected Python version: !PYVER!

echo !PYVER!|findstr /b /c:"3.10" /c:"3.11" /c:"3.12">nul
if errorlevel 1 (
    echo [ERROR] Python !PYVER! detected. TensorFlow 2.16 requires Python 3.10-3.12.
    echo         Install Python 3.10, 3.11, or 3.12 and run this launcher again.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment in venv\ ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

call "venv\Scripts\activate.bat"

echo [INFO] Installing / checking requirements ...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. See log above.
    pause
    exit /b 1
)

echo.
echo [INFO] Running environment check ...
python setup\check_env.py
echo.

if not exist "models\alphabet_model.keras" (
    echo [WARN ] alphabet_model.keras not found.
    if exist "dataset\alphabet" (
        echo [INFO ] dataset\alphabet found. Running preprocessing + training for alphabet.
        python src\preprocess_alphabet.py
        if not errorlevel 1 python src\train_alphabet.py --quick
    )
)
if not exist "models\word_model.keras" (
    echo [WARN ] word_model.keras not found.
    if exist "dataset\words" (
        echo [INFO ] dataset\words found. Running preprocessing + training for words (may take a while).
        python src\preprocess_words.py
        if not errorlevel 1 python src\train_words.py --quick
    )
)

echo.
echo [INFO] Launching Streamlit UI ...
streamlit run app.py
pause
endlocal
