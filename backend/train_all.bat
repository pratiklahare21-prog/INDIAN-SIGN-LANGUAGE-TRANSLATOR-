@echo off
echo Training all ISL Sign2Speech models...
cd /d "%~dp0"

echo.
echo [1/3] Training alphabet model...
python src\train_alphabet.py
if errorlevel 1 (
    echo ERROR: Alphabet training failed
    pause
    exit /b 1
)

echo.
echo [2/3] Training word model...
python src\train_words.py
if errorlevel 1 (
    echo ERROR: Word training failed
    pause
    exit /b 1
)

echo.
echo [3/3] Training sentence model...
python src\train_sentences.py
if errorlevel 1 (
    echo ERROR: Sentence training failed
    pause
    exit /b 1
)

echo.
echo === All models trained successfully! ===
pause
