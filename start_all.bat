@echo off
chcp 65001 >nul
title 对话质量分类系统 - 一键启动

set "PROJECT_DIR=C:\Users\tianxuan3plus\Desktop\AI 标注 + 文本分类"
set "PYTHON=D:/Python/pycharm/python"

echo ================================================
echo   对话质量分类系统
echo   启动中...
echo ================================================

echo   [1] 正在启动 推理 API (端口 8000)...
start "推理API" cmd /k "cd /d %PROJECT_DIR% && %PYTHON% api_server.py --port 8000 && pause"

echo   [2] 正在启动 LabelStudio (端口 8080)...
start "LabelStudio" cmd /k "cd /d %PROJECT_DIR% && ls_venv\Scripts\activate && label-studio start"

echo   [3] 正在启动 ML Backend (端口 9090)...
start "ML_Backend" cmd /k "cd /d %PROJECT_DIR% && %PYTHON% ml_backend.py --port 9090 && pause"

echo.
echo ================================================
echo   启动完成！等10秒让模型加载完...
echo ================================================
echo.
echo   模型加载好后，用浏览器打开：
echo.
echo   演示页面:     %PROJECT_DIR%\demo.html
echo   Swagger文档:  http://localhost:8000/docs
echo   LabelStudio:  http://localhost:8080
echo.
echo ================================================
timeout /t 5 /nobreak >nul
start "" "%PROJECT_DIR%\demo.html"
