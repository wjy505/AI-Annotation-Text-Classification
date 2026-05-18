@echo off
title Dialog Quality Classifier

set PROJECT_DIR=C:\Users\tianxuan3plus\Desktop\AI 标注 + 文本分类
set PY=D:\Python\pycharm\python.exe

echo ================================================
echo   Starting Dialog Quality Classifier...
echo ================================================

cd /d "%PROJECT_DIR%"

echo   [1] Inference API (port 8000)...
start "API" /D "%PROJECT_DIR%" %PY% api_server.py --port 8000

echo   [2] LabelStudio (port 8080)...
start "LabelStudio" /D "%PROJECT_DIR%" cmd /k "ls_venv\Scripts\activate.bat && label-studio start"

echo   [3] ML Backend (port 9090)...
start "MLBackend" /D "%PROJECT_DIR%" %PY% ml_backend.py --port 9090

echo ================================================
echo   Wait 15s for model to load, then open demo.html
echo     Swagger:      http://localhost:8000/docs
echo     LabelStudio:  http://localhost:8080
echo ================================================
timeout /t 15 /nobreak >nul
start "" "%PROJECT_DIR%\demo.html"
echo Done.
pause
