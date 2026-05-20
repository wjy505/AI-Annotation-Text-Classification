"""
ml_backend.py — LabelStudio ML Backend 预标注服务。

启动:
  python ml_backend.py --port 9090

在 LabelStudio 中对接:
  Settings → Machine Learning → Add Model
  → URL: http://localhost:9090
  → 勾选 "Use for interactive preannotations"

LabelStudio 把任务发给本服务 → 模型自动预测 → 标注员只需审核修正
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# 网络配置
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_backend")

# ═══════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════

MODEL_DIR = "checkpoints/best_model"
MAX_LEN = 256

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.eval()

    label_map_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        id2label = {int(k): v for k, v in mp["id2label"].items()}
    else:
        id2label = model.config.id2label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, id2label, device

model, tokenizer, id2label, device = load_model(MODEL_DIR)
MODEL_VERSION = "1.0.0"
logger.info(f"  模型已加载: {MODEL_DIR}  |  设备: {device}  |  类别: {list(id2label.values())}")

# ═══════════════════════════════════════════════════
#  FastAPI
# ═══════════════════════════════════════════════════

app = FastAPI(
    title="ML Backend for LabelStudio",
    description="对话质量自动预标注：接收 LabelStudio 任务，返回模型预测结果",
    version=MODEL_VERSION,
)

# ─── 配置（labeling_config.xml 中的 tag name 对应关系）───
# 这些值与 labeling_config.xml 中的 name 属性保持一致
FROM_NAME_CHOICES = "usefulness"   # <Choices name="usefulness">
FROM_NAME_TEXTAREA = "error_type"  # <TextArea name="error_type">
TO_NAME = "user_msg"               # <Text name="user_msg">


def predict_single(text: str) -> tuple:
    """预测单条文本，返回 (label, confidence, top_scores)。"""
    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    label = id2label[top_idx]
    confidence = float(probs[top_idx])
    # 按概率降序排列所有类别
    scores = []
    for i in np.argsort(probs)[::-1]:
        scores.append({"label": id2label[i], "score": float(probs[i])})
    return label, confidence, scores


def build_annotation(label: str, confidence: float):
    """构建 LabelStudio 标注结果对象。

    每个任务的返回格式必须是:
    {
      "result": [
        {"from_name": "标签名", "to_name": "文本名", "type": "choices", "value": {"choices": [...]}},
        ...
      ],
      "score": 置信度
    }
    """
    return {
        "model_version": MODEL_VERSION,
        "result": [
            {
                "from_name": FROM_NAME_CHOICES,
                "to_name": TO_NAME,
                "type": "choices",
                "value": {
                    "choices": [label]
                },
            }
        ],
        "score": confidence,
    }


# ═══════════════════════════════════════════════════
#  ML Backend API（LabelStudio 协议）
# ═══════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "up", "model_version": MODEL_VERSION, "device": device}

@app.api_route("/setup", methods=["GET", "POST"])
def setup():
    """返回模型支持的标注类型，供 LabelStudio 自动匹配。"""
    return {
        "model_version": MODEL_VERSION,
        "model_class": "dialog-quality-classifier",
        "description": "对话质量有用/无用分类模型",
        "capabilities": {
            "predictions": True,
            "async_predictions": False,
        },
    }

@app.post("/predict")
async def predict_endpoint(request: Request):
    """LabelStudio ML Backend 核心接口：接收任务列表，返回预测标注。

    请求格式:
      {"tasks": [{"id": 1, "data": {"user": "...", "assistant": "..."}}, ...]}

    返回格式:
      {"results": [{"id": 1, "result": [...], "score": 0.85}, ...]}
    """
    body = await request.json()
    logger.info(f"收到预测请求: {len(body.get('tasks', []))} 个任务")
    if body.get("tasks"):
        logger.info(f"  首个任务: {json.dumps(body['tasks'][0], ensure_ascii=False)}")
    tasks = body.get("tasks", [])

    results = []
    for task in tasks:
        task_id = task.get("id", None)
        data = task.get("data", {})

        # 提取 user 和 assistant 文本
        user = data.get("user", "")
        assistant = data.get("assistant", "")

        if not user or not assistant:
            results.append({
                "id": task_id,
                "result": [],
                "score": 0.0,
            })
            continue

        # 合并文本并预测
        text = f"用户: {user}\n助手: {assistant}"
        label, confidence, scores = predict_single(text)

        annotation = build_annotation(label, confidence)
        annotation["id"] = task_id

        logger.info(f"  task {task_id}: {label} ({confidence:.3f})  |  user: {user[:40]}...")
        results.append(annotation)

    return {"results": results}


@app.get("/is_ready")
def is_ready():
    return {"is_ready": True}

@app.post("/teardown")
def teardown():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════
#  自定义：批量预测（独立端，用于调试）
# ═══════════════════════════════════════════════════

class BatchPredictRequest(BaseModel):
    tasks: list[dict] = Field(..., description="任务列表 [{\"id\": ..., \"data\": {\"user\": ..., \"assistant\": ...}}]")

@app.post("/debug/predict")
def debug_predict(req: BatchPredictRequest):
    """独立调试端：不需要对接 LabelStudio，直接 POST 任务列表即可。"""
    results = []
    for task in req.tasks:
        user = task.get("data", {}).get("user", "")
        assistant = task.get("data", {}).get("assistant", "")
        text = f"用户: {user}\n助手: {assistant}"
        label, confidence, scores = predict_single(text)
        results.append({
            "id": task.get("id"),
            "prediction": label,
            "confidence": confidence,
            "all_scores": scores,
        })
    return {"results": results}


# ═══════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LabelStudio ML Backend 预标注服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=9090, help="端口")
    parser.add_argument("--model", default=MODEL_DIR, help="模型目录")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
