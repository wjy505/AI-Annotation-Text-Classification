# 对话质量分类系统

基于 LabelStudio + PyTorch + FastAPI + Docker 的完整 AI 标注与分类系统。

## 项目概述

自动判断 AI 助手的回复质量（有用 / 无用），支持从数据标注到模型部署的完整 MLOps 流水线。

## 快速开始

你在 AI 公司负责对话质量评估，手头有三样东西：
1. 一台 Windows 电脑（Python 3.10 已装好）
2. 你已经标注过几百条数据的 LabelStudio
3. 你想部署一个 API 让其他服务调用

### 第一次用

```bash
# 1. 装依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate scikit-learn numpy tqdm fastapi uvicorn

# 2. 下载中文预训练模型（国内设镜像）
set HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import AutoModel; AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')"

# 3. 训练（用生成好的 110 条数据先跑通）
python generate_training_data.py
python quality_check.py -i annotations.json -oc cleaned.jsonl -ox cls.jsonl --label-studio --balance-check
python train.py --data cls.jsonl --epochs 5 --batch-size 16 --no-cuda

# 4. 启动 API
python api_server.py --port 8000

# 5. 测试
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"用户: 问题\n助手: 回答\"}"
```

### 接入你自己的数据

```bash
# 1. 把原始对话转成 LabelStudio 格式
python bootstrap_data.py -i 你的数据.csv -o 待标注.json

# 2. 启动 LabelStudio
python -m venv ls_env && ls_env\Scripts\activate
pip install label-studio
label-studio start
# → 浏览器 http://localhost:8080 → 创建项目 → 粘贴 labeling_config.xml → 导入 待标注.json

# 3. 人工标注（Web UI 操作）
# → 逐条标注 useful/useless → Export 导出 JSON → 放到项目目录改名 annotations.json

# 4. 一键跑完余下流程
python quality_check.py -i annotations.json -oc cleaned.jsonl -ox cls.jsonl --label-studio --balance-check
python convert_to_deepseek.py cleaned.jsonl -o train.jsonl
python train.py --data cls.jsonl --epochs 5 --batch-size 16 --no-cuda
python api_server.py --port 8000
```

### Docker 部署

```bash
docker build -t dialog-classifier .
docker run -d -p 8000:8000 --name dialog-api dialog-classifier
# 全栈
docker-compose up -d
```

## 项目结构

```
├── api_server.py           # FastAPI 推理服务（/predict /batch /health）
├── train.py                # HuggingFace Trainer 训练脚本
├── predict.py              # 命令行推理
├── quality_check.py        # 质检清洗 + 统计报告
├── convert_to_deepseek.py  # → DeepSeek SFT JSONL
├── bootstrap_data.py       # 原始数据 → LabelStudio 导入格式
├── generate_training_data.py # 模拟数据生成（110条）
├── export_annotations.py   # LabelStudio SDK 导出
├── labeling_config.xml     # 标注界面 UI 配置
├── Dockerfile / docker-compose.yml
├── docs/                   # 需求/设计/操作文档
└── devlog/                 # 开发日志
```

## API 文档

启动服务后访问 `http://localhost:8000/docs`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /predict | 单条预测 |
| POST | /batch | 批量预测（≤1000条） |

## 模型性能

| 指标 | 数值 |
|------|------|
| 模型 | hfl/chinese-roberta-wwm-ext |
| 训练数据 | 110 条（70 useful / 40 useless） |
| 准确率 | 86.4% |
| F1-macro | 85.6% |
| 推理延迟 | <100ms (CPU) |

## 技术栈

Python 3.10 · PyTorch 2.x · HuggingFace Transformers · FastAPI · Docker · LabelStudio · scikit-learn
