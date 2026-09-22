@echo off
echo ================================================
echo ISL Sign2Speech - Starting Application
echo ================================================
cd /d "%~dp0"
if exist "frontend\venv\Scripts\python.exe" (
    frontend\venv\Scripts\python.exe -m streamlit run frontend\app.py
) else (
    python -m streamlit run frontend\app.py
)
