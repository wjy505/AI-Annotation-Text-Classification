"""一键启动所有服务。双击此文件或运行: python start_all.py"""

import subprocess, os, time, webbrowser

PROJECT_DIR = r"C:\Users\tianxuan3plus\Desktop\AI 标注 + 文本分类"
PYTHON = "D:/Python/pycharm/python"

os.chdir(PROJECT_DIR)

print("=" * 50)
print("  启动对话质量分类系统...")
print("=" * 50)

# 启动 API
print("  [1] 推理 API (8000)...")
subprocess.Popen([PYTHON, "api_server.py", "--port", "8000"],
    cwd=PROJECT_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)

# 启动 ML Backend
print("  [2] ML Backend (9090)...")
subprocess.Popen([PYTHON, "ml_backend.py", "--port", "9090"],
    cwd=PROJECT_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)

# 启动 LabelStudio
print("  [3] LabelStudio (8080)...")
subprocess.Popen(["cmd", "/k", r"ls_venv\Scripts\activate.bat && label-studio start"],
    cwd=PROJECT_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)

print()
print("=" * 50)
print("  等待 15 秒让模型加载...")
print("  Swagger:      http://localhost:8000/docs")
print("  LabelStudio:  http://localhost:8080")
print("=" * 50)

time.sleep(15)
webbrowser.open(os.path.join(PROJECT_DIR, "demo.html"))
print("  演示页面已打开。")
input("  按回车关闭此窗口(服务会继续运行)...")
