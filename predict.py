"""
predict.py — 文本分类推理脚本。

用法:
  python predict.py -m checkpoints/best_model -t "用户: 你好\n助手: 你好！有什么可以帮你的？"
  python predict.py -m checkpoints/best_model -f input.txt
  python predict.py -m checkpoints/best_model -f cls.jsonl --show-top-k 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 国内网络自动使用 HF 镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_dir: str):
    """加载微调后的模型、分词器、标签映射。"""
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # 加载 label 映射
    label_map_path = Path(model_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        id2label = {int(k): v for k, v in mp["id2label"].items()}
    else:
        id2label = model.config.id2label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, id2label, device


def predict_single(model, tokenizer, id2label, device, text: str, max_len=256):
    """预测单条文本，返回 (label, confidence) 和 top-k 列表。"""
    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    label = id2label[top_idx]
    confidence = float(probs[top_idx])

    top_k = [
        (id2label[i], float(probs[i]))
        for i in np.argsort(probs)[::-1]
    ]
    return label, confidence, top_k


def main():
    parser = argparse.ArgumentParser(description="文本分类推理")
    parser.add_argument("-m", "--model", required=True, help="模型目录路径")
    parser.add_argument("-t", "--text", help="单条文本输入")
    parser.add_argument("-f", "--file", help="文本文件或 JSONL 文件")
    parser.add_argument("--show-top-k", type=int, default=3, help="显示 top-k 类别")
    parser.add_argument("--max-len", type=int, default=256, help="最大序列长度")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("错误: 请指定 --text 或 --file", file=sys.stderr)
        sys.exit(1)

    # 加载模型
    model, tokenizer, id2label, device = load_model(args.model)
    model.to(device)

    # 收集输入文本
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"错误: 文件不存在: {args.file}", file=sys.stderr)
            sys.exit(1)
        raw = path.read_text(encoding="utf-8")
        # 尝试作为 JSONL 解析
        try:
            items = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
            texts = [item["text"] for item in items]
        except (json.JSONDecodeError, KeyError):
            # 作为普通文本文件，按行处理
            texts = [line.strip() for line in raw.strip().split("\n") if line.strip()]

    # 逐条预测
    for i, text in enumerate(texts):
        label, conf, top_k = predict_single(
            model, tokenizer, id2label, device, text, args.max_len
        )
        if len(texts) == 1:
            print(f"\n  预测结果: {label}  (置信度: {conf:.4f})\n")
            print(f"  Top-{min(args.show_top_k, len(top_k))} 类别:")
            for j, (lbl, p) in enumerate(top_k[:args.show_top_k]):
                bar = "█" * int(p * 30)
                print(f"    {j+1}. {lbl:<12s}  {p:.4f}  {bar}")
        else:
            print(f"  [{i+1:>4d}] {label:<12s} ({conf:.4f})")
            if args.show_top_k > 1:
                others = "  ".join(f"{l}={p:.3f}" for l, p in top_k[1:args.show_top_k])
                print(f"         {others}")


if __name__ == "__main__":
    main()
