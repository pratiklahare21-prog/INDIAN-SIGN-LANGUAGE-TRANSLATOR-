@echo off
echo ================================================
echo ISL Sign2Speech - Setup and Run
echo ================================================

cd /d "%~dp0"

:: Check if backend venv exists
if not exist "backend\venv\Scripts\python.exe" (
    echo [1/4] Creating backend virtual environment...
    cd backend
    python -m venv venv
    cd ..
) else (
    echo [1/4] Backend venv exists, skipping...
)

:: Check if frontend venv exists
if not exist "frontend\venv\Scripts\python.exe" (
    echo [2/4] Creating frontend virtual environment...
    cd frontend
    python -m venv venv
    cd ..
) else (
    echo [2/4] Frontend venv exists, skipping...
)

:: Install backend dependencies
echo [3/4] Installing backend dependencies...
cd backend
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
call deactivate
cd ..

:: Install frontend dependencies
echo [4/4] Installing frontend dependencies...
cd frontend
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

echo.
echo ================================================
echo Setup Complete! Starting Streamlit...
echo ================================================
echo.
echo The app will open at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

:: Start Streamlit
streamlit run app.py
