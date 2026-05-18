@echo off
chcp 65001 >nul

set "PROJECT_DIR=C:\Users\tianxuan3plus\Desktop\AI 标注 + 文本分类"

echo ================================================
echo   对话质量分类系统
echo   启动中...
echo ================================================

cd /d "%PROJECT_DIR%"

echo   [1] 推理API (端口8000)...
start "API" /D "%PROJECT_DIR%" D:\Python\pycharm\python.exe api_server.py --port 8000

echo   [2] LabelStudio (端口8080)...
start "LabelStudio" /D "%PROJECT_DIR%" cmd /k "ls_venv\Scripts\activate.bat && label-studio start"

echo   [3] ML Backend (端口9090)...
start "MLBackend" /D "%PROJECT_DIR%" D:\Python\pycharm\python.exe ml_backend.py --port 9090

echo.
echo   15秒后自动打开演示页面...
timeout /t 15 /nobreak >nul
start "" "%PROJECT_DIR%\demo.html"
echo   完毕。
pause
