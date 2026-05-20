# 项目名称

**AI-Annotation-Text-Classification · 对话质量分类系统**

GitHub: https://github.com/wjy505/AI-Annotation-Text-Classification

---

## 项目介绍

本项目是一个面向 AI 对话质量评估的完整 MLOps 系统。从数据标注、质检清洗、模型训练到 API 部署和 Docker 容器化，覆盖了机器学习工程的全链路。

核心功能：对 AI 助手与用户的对话进行自动质量判别（有用 / 无用），并提供事实错误识别能力。系统采用人机协同的数据飞轮架构——模型自动预标注，人工只需复核修正，新标注数据回流训练，模型迭代提升。

项目已开源发布在 GitHub，包含 9 个 Python 工程模块、3 个 Docker 配置文件、4 份技术文档，代码总行数超过 2000 行。

---

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 预训练模型 | HuggingFace Transformers · hfl/chinese-roberta-wwm-ext |
| 深度学习 | PyTorch 2.x · Accelerate · scikit-learn |
| API 服务 | FastAPI · Uvicorn · Pydantic · Swagger 自动文档 |
| 数据标注 | LabelStudio 1.23 · ML Backend 预标注协议 |
| 数据管道 | JSONL 流式处理 · MD5 去重 · 标签平衡分析 |
| 容器化 | Docker · docker-compose · python:3.10-slim |
| 版本控制 | Git · GitHub |
| 语言 | Python 3.10 |

---

## 技术亮点

**1. 全链路 MLOps 工程化**

不同于仅停留在 Jupyter Notebook 的实验性项目，本系统实现了从数据到模型再到服务的完整闭环。标注数据经质量检查流水线自动清洗后，一键训练、一键部署，全程可复现。

**2. 数据飞轮架构**

基于 LabelStudio ML Backend 协议实现模型预标注服务，将人工标注效率从零标注提升为审核修正模式。新标注数据可自动回流训练集，形成 "数据 → 模型 → 预标注 → 人工审核 → 新数据" 的正向循环。

**3. 健壮的数据质量系统**

自研质检脚本包含 8 项自动检查：空值过滤、非法角色剔除、Unicode 转义修复、MD5 去重、标签平衡性可视化、LabelStudio 嵌套格式自动解析，输出完整的质检报告。

**4. 国内网络环境适配**

针对国内网络环境进行了全面适配：自动切换 HuggingFace 镜像、PyPI 清华源、Docker Registry 镜像加速、PyTorch CPU 索引，确保开箱即用。

**5. 生产级 API 设计**

FastAPI 服务包含健康检查、单条预测、批量预测三个端点，Pydantic 类型校验，自动生成 Swagger 交互式文档。支持 Docker 容器化部署，docker-compose 一键启动全栈服务（API + ML Backend + LabelStudio）。

**6. 模型性能**

基于 110 条标注数据对 RoBERTa-wwm-ext 进行微调，验证集准确率 86.4%，F1-macro 85.6%，Macro F1 接近 86%。在 CPU 环境下推理延迟低于 100ms。

---

## 项目结构

```
├── api_server.py           FastAPI 推理服务
├── ml_backend.py           LabelStudio ML Backend 预标注
├── train.py                HuggingFace Trainer 训练脚本
├── predict.py              命令行推理
├── quality_check.py        质检清洗 + 统计报告
├── convert_to_deepseek.py  DeepSeek SFT JSONL 转换
├── bootstrap_data.py       原始数据 → LabelStudio 导入
├── generate_training_data.py 模拟训练数据生成
├── export_annotations.py   LabelStudio SDK 导出
├── batch_prelabel.py       批量预标注
├── Dockerfile / docker-compose.yml
├── labeling_config.xml     标注界面配置
├── demo.html               在线演示页面
├── docs/                   需求/设计/操作文档
└── devlog/                 开发日志
```
