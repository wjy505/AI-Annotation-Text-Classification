# CLAUDE.md — 对话文本标注项目指引

## 项目概述

本项目用于搭建基于 LabelStudio 的对话文本标注平台，对用户消息与 AI 助手回复的质量进行人工标注（有用/无用 + 错误类型）。

## 标准文件路径

| 路径 | 用途 |
|------|------|
| `CLAUDE.md` | 本文件 — 项目指引 |
| `sample_conversations.json` | LabelStudio 导入用的示例标注数据 |
| `labeling_config.xml` | LabelStudio 标注界面 XML 配置 |
| `export_annotations.py` | 基于 label-studio-sdk 的导出脚本 |
| `convert_to_deepseek.py` | 标注结果 → DeepSeek 微调 JSONL 转换脚本 |
| `quality_check.py` | 质检 + 自动修复脚本，输出清洗后 JSONL + 质检报告 |
| `train.py` | 文本分类训练脚本（PyTorch + HuggingFace） |
| `predict.py` | 分类推理脚本（单条/批量） |
| `bootstrap_data.py` | 原始对话 → LabelStudio 导入格式转换 |
| `api_server.py` | FastAPI 推理服务（单条/批量预测） |
| `ml_backend.py` | LabelStudio ML Backend 预标注服务 |
| `Dockerfile` | API 服务镜像构建 |
| `docker-compose.yml` | 全栈部署（API + ML Backend + LabelStudio） |
| `requirements.txt` | Python 依赖清单（torch, transformers, sklearn） |
| `devlog/` | 每日开发日志目录 |
| `devlog/YYYY-MM-DD.md` | 当日开发日志（按日期命名） |
| `docs/requirements.md` | 项目需求文档 |
| `docs/design.md` | 技术设计与架构 |
| `docs/procedures.md` | 完整操作步骤手册 |
| `scripts/` | 工具脚本目录（预留） |

## 环境说明

项目使用两个 Python 环境分工：

| 用途 | Python | 激活方式 |
|------|--------|---------|
| 标注 (LabelStudio) | ls_venv/ | `ls_venv\Scripts\activate` |
| 训练/推理/数据处理 | `D:/Python/pycharm/python` (3.10.10) | 直接调用完整路径 |

- **标注工具**: label-studio 1.23.0 (ls_venv)
- **训练框架**: torch 2.12.0+cpu + transformers 5.8.1 + accelerate (系统 Python)
- **服务地址**: http://localhost:8080
- **数据存储**: `~/.label-studio/` (SQLite)
- **HF 镜像**: `set HF_ENDPOINT=https://hf-mirror.com`（国内必设）
- **devlog 提醒**: 每个工作日 18:03 自动提醒

## 常用命令

```bash
# ── LabelStudio（用 ls_venv）──
ls_venv\Scripts\activate
label-studio start
# 然后 Web UI Export 下载 JSON 即可（无需 SDK Token）

# ── 数据处理（用系统 Python）──
D:/Python/pycharm/python bootstrap_data.py -i raw_chats.json -o ls_import.json
D:/Python/pycharm/python quality_check.py -i annotations.json -oc cleaned.jsonl -ox cls.jsonl --label-studio --balance-check
D:/Python/pycharm/python convert_to_deepseek.py cleaned.jsonl -o train.jsonl

# ── 训练 & 推理（用系统 Python）──
set HF_ENDPOINT=https://hf-mirror.com
D:/Python/pycharm/python train.py --data cls.jsonl --epochs 3 --batch-size 16 --no-cuda
D:/Python/pycharm/python predict.py -m checkpoints/best_model -t "用户: 你好\n助手: 你好！"
D:/Python/pycharm/python predict.py -m checkpoints/best_model -f cls.jsonl

# 启动推理 API 服务
D:/Python/pycharm/python api_server.py --port 8000
# → 访问 http://localhost:8000/docs 查看交互式文档
# → curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"...\"}"

# Docker 部署
docker build -t dialog-classifier .
docker run -p 8000:8000 dialog-classifier
# 或全栈启动（API + LabelStudio）
docker-compose up -d
```

## 开发工作流

1. 修改标注配置 → 更新 `labeling_config.xml` → 在 LabelStudio Web UI 中同步
2. 新增示例数据 → 更新 `sample_conversations.json` → 导入项目
3. 导出结果 → 使用 `export_annotations.py` 或 Web UI Export
4. 每日结束前 → 更新 `devlog/YYYY-MM-DD.md`

## devlog 约定

- 每天工作结束前更新当日开发日志
- 文件名格式: `YYYY-MM-DD.md`
- 内容包含: 今日完成、待办事项、遇到的问题

## docs 约定

- `requirements.md` — 需求变更时更新
- `design.md` — 架构 / 配置变更时更新
- `procedures.md` — 流程变更或新增工具时更新
