"""
api_server.py — 对话质量分类推理 API 服务。

启动:
  python api_server.py
  python api_server.py --port 8000 --model checkpoints/best_model

端点:
  GET  /            — API 文档页（交互式 Swagger）
  GET  /health      — 健康检查
  POST /predict     — 单条预测
  POST /batch       — 批量预测
"""

import os
import sys
from typing import Optional

# 国内网络自动使用 HF 镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ═══════════════════════════════════════════════════
#  全局模型加载（启动时一次）
# ═══════════════════════════════════════════════════

MODEL_DIR = "checkpoints/best_model"
MAX_LEN = 256

def load_model(model_dir: str):
    """加载模型、分词器、标签映射。"""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.eval()

    # 标签映射
    label_map_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(label_map_path):
        import json
        with open(label_map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        id2label = {int(k): v for k, v in mp["id2label"].items()}
    else:
        id2label = model.config.id2label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, id2label, device

model, tokenizer, id2label, device = load_model(MODEL_DIR)
print(f"  模型已加载: {MODEL_DIR}  |  设备: {device}  |  类别: {list(id2label.values())}")

# ═══════════════════════════════════════════════════
#  FastAPI
# ═══════════════════════════════════════════════════

app = FastAPI(
    title="对话质量分类 API",
    description="对用户-AI对话进行有用/无用二分类，支持单条和批量预测",
    version="1.0.0",
)

# ─── 演示页面 ───
import pathlib

@app.get("/demo")
def demo():
    demo_html = pathlib.Path(__file__).parent / "demo.html"
    from fastapi.responses import HTMLResponse
    return HTMLResponse(demo_html.read_text(encoding="utf-8"))

# ─── 请求/响应模型 ───

class PredictRequest(BaseModel):
    text: str = Field(..., description="合并后的对话文本（格式: 用户: xxx\\n助手: xxx）",
                      example="用户: Python和Java的区别？\n助手: Python动态类型，Java静态类型。")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "用户: Redis适合什么场景？\n助手: Redis适用于缓存、消息队列、会话管理等高性能场景。"
            }
        }

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., description="对话文本列表", min_length=1, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "用户: 什么是RESTful API？\n助手: REST是Representational State Transfer的缩写...",
                    "用户: 什么是递归？\n助手: 递归嘛...就是自己调用自己吧，大概就是这样。"
                ]
            }
        }

class PredictResult(BaseModel):
    label: str = Field(..., description="预测类别 (useful / useless)")
    confidence: float = Field(..., description="置信度 (0-1)")
    top_k: list[dict] = Field(..., description="所有类别及其概率")

class BatchResult(BaseModel):
    results: list[PredictResult]
    total: int

# ─── 核心预测函数 ───

def predict(text: str) -> dict:
    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_k = sorted(
        [{"label": id2label[i], "probability": round(float(probs[i]), 4)} for i in range(len(probs))],
        key=lambda x: x["probability"], reverse=True,
    )
    return {
        "label": id2label[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "top_k": top_k,
    }

# ─── 端点 ───

@app.get("/")
def root():
    return {"service": "对话质量分类 API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_DIR, "device": device}

@app.post("/predict", response_model=PredictResult)
def predict_single(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="文本不能为空")
    return predict(req.text)

@app.post("/batch", response_model=BatchResult)
def predict_batch(req: BatchRequest):
    results = []
    for text in req.texts:
        if not text.strip():
            results.append(PredictResult(
                label="error", confidence=0.0,
                top_k=[{"label": "error", "probability": 0.0}],
            ))
        else:
            results.append(PredictResult(**predict(text)))
    return BatchResult(results=results, total=len(results))


# ═══════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对话质量分类 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    parser.add_argument("--model", default=MODEL_DIR, help="模型目录")
    args = parser.parse_args()

    MODEL_DIR = args.model
    model, tokenizer, id2label, device = load_model(MODEL_DIR)
    print(f"  模型已加载: {MODEL_DIR}  |  设备: {device}")

    uvicorn.run(app, host=args.host, port=args.port)
