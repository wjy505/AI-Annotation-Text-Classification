"""
train.py — 文本分类训练脚本（PyTorch + HuggingFace Transformers）。

用法:
  python train.py --data cls.jsonl --epochs 3 --batch-size 16
  python train.py --data cls.jsonl --model hfl/chinese-roberta-wwm-ext --epochs 5 --lr 2e-5
  python train.py --data cls.jsonl --model hfl/chinese-bert-wwm-ext --epochs 3
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# 国内网络自动使用 HF 镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════
#  数据集
# ═══════════════════════════════════════════════════

class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data(filepath: str) -> tuple:
    """读取 JSONL 文件，返回 (texts: list[str], labels: list[int], label_map: dict)。"""
    texts, raw_labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append(item["text"])
            raw_labels.append(item["label"])

    # 构建 label → id 映射
    unique = sorted(set(raw_labels))
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    labels = [label2id[l] for l in raw_labels]

    print(f"  数据加载: {len(texts)} 条, {len(unique)} 个类别: {unique}")
    return texts, np.array(labels), label2id, id2label


# ═══════════════════════════════════════════════════
#  指标
# ═══════════════════════════════════════════════════

def compute_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="文本分类训练")
    parser.add_argument("--data", required=True, help="JSONL 输入文件（含 text/label 字段）")
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext",
                        help="HuggingFace 模型名（默认 hfl/chinese-roberta-wwm-ext）")
    parser.add_argument("--output", default="./checkpoints", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max-len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--no-cuda", action="store_true", help="强制使用 CPU")
    parser.add_argument("--no-stratify", action="store_true", help="验证集不按标签分层")
    args = parser.parse_args()

    # ── 设备 ──
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"  设备: {device}")

    # ── 加载数据 ──
    if not Path(args.data).exists():
        print(f"错误: 输入文件不存在: {args.data}", file=sys.stderr)
        sys.exit(1)

    texts, labels, label2id, id2label = load_data(args.data)
    n_classes = len(label2id)
    print(f"  Label映射: {label2id}")

    # ── 划分训练/验证集 ──
    stratify = labels if not args.no_stratify and n_classes <= len(labels) * args.val_split else None
    xt, xv, yt, yv = train_test_split(
        texts, labels, test_size=args.val_split, random_state=args.seed,
        stratify=stratify,
    )
    print(f"  训练集: {len(xt)}, 验证集: {len(xv)}")

    # ── 加载模型 / 分词器 ──
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # hfl/roberta 系列没有 pad_token，用 eos_token 或 [PAD]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, local_files_only=True,
    )
    model.to(device)

    # ── 构建 Dataset ──
    train_ds = TextClsDataset(xt, yt, tokenizer, max_len=args.max_len)
    val_ds = TextClsDataset(xv, yv, tokenizer, max_len=args.max_len)

    # ── Trainer ──
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(device == "cuda"),
        dataloader_drop_last=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 训练 ──
    print("\n  开始训练...")
    trainer.train()

    # ── 验证集评估 ──
    print("\n  验证集评估:")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")

    # ── 类别级报告 ──
    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\n" + classification_report(
        yv, y_pred,
        labels=list(range(n_classes)),
        target_names=[str(id2label[i]) for i in range(n_classes)],
        zero_division=0,
    ))

    # ── 保存模型 ──
    save_path = os.path.join(args.output, "best_model")
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n  最佳模型已保存至: {save_path}")

    # 保存 label 映射
    with open(os.path.join(save_path, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
