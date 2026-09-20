@echo off
echo Preprocessing all ISL Sign2Speech datasets...
cd /d "%~dp0"

echo.
echo [1/3] Preprocessing alphabet dataset...
python src\preprocess_alphabet.py
if errorlevel 1 (
    echo ERROR: Alphabet preprocessing failed
    pause
    exit /b 1
)

echo.
echo [2/3] Preprocessing word dataset...
python src\preprocess_words.py
if errorlevel 1 (
    echo ERROR: Word preprocessing failed
    pause
    exit /b 1
)

echo.
echo [3/3] Preprocessing sentence dataset...
python src\preprocess_sentences.py
if errorlevel 1 (
    echo ERROR: Sentence preprocessing failed
    pause
    exit /b 1
)

echo.
echo === All datasets preprocessed successfully! ===
pause
