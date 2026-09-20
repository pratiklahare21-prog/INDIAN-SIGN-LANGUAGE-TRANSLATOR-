@echo off
echo ================================================
echo ISL Sign2Speech - Starting Application
echo ================================================
cd /d "%~dp0"
.\venv\Scripts\python.exe -m streamlit run frontend\app.py
