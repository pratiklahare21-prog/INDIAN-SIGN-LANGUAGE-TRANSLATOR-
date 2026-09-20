@echo off
xcopy /Y /I src\*.py backend\src\
copy /Y config.py backend\config.py
