@echo off
chcp 65001 >nul
title 对话质量分类系统 - 一键启动

echo ================================================
echo   对话质量分类系统
echo   启动中...
echo ================================================
echo.
echo   [1] 正在启动 LabelStudio (端口 8080)...
start "LabelStudio 标注平台" cmd /k "cd /d %~dp0 && ls_venv\Scripts\activate && label-studio start"

echo   [2] 正在启动 推理 API (端口 8000)...
start "推理 API" cmd /k "cd /d %~dp0 && D:/Python/pycharm/python api_server.py --port 8000"

echo   [3] 正在启动 ML Backend (端口 9090)...
start "ML Backend 预标注" cmd /k "cd /d %~dp0 && D:/Python/pycharm/python ml_backend.py --port 9090"

echo.
echo ================================================
echo   启动完成！3 个服务窗口已打开
echo ================================================
echo.
echo   请在浏览器中访问：
echo.
echo   推理 API 调试:  http://localhost:8000/docs
echo   LabelStudio:  http://localhost:8080
echo   ML Backend:   http://localhost:9090/health
echo.
echo   按任意键关闭此窗口... (服务会继续运行)
echo ================================================
pause >nul
