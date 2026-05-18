@echo off
chcp 65001 >nul
title 对话质量分类系统 - 一键启动

set "PROJECT_DIR=C:\Users\tianxuan3plus\Desktop\AI 标注 + 文本分类"
set "PY=D:/Python/pycharm/python.exe"

echo ================================================
echo   对话质量分类系统
echo   启动中...
echo ================================================

echo   [1] 正在启动 推理 API ^(端口 8000^)...
start "推理API" cmd /c "cd /d %PROJECT_DIR% && %PY% api_server.py --port 8000"

echo   [2] 正在启动 LabelStudio ^(端口 8080^)...
start "LabelStudio" cmd /c "cd /d %PROJECT_DIR% && call ls_venv\Scripts\activate.bat && label-studio start"

echo   [3] 正在启动 ML Backend ^(端口 9090^)...
start "ML_Backend" cmd /c "cd /d %PROJECT_DIR% && %PY% ml_backend.py --port 9090"

echo.
echo ================================================
echo   启动完成！请等待约 15 秒让模型加载...
echo ================================================
echo.
echo   模型加载好后：
echo     - 演示页面  ^> 双击 demo.html
echo     - Swagger   ^> http://localhost:8000/docs
echo     - LabelStudio ^> http://localhost:8080
echo.
echo ================================================
timeout /t 10 /nobreak >nul
start "" "%PROJECT_DIR%\demo.html"
