"""
bootstrap_data.py — 将原始对话数据批量转换为 LabelStudio 导入格式。

支持输入:
  - JSON/JSONL（每行含 user + assistant 字段）
  - CSV（含 user + assistant 列）
  - 纯文本文件（每 2 行为一组对话：奇数行=user，偶数行=assistant）

输出: LabelStudio 可导入的 JSON 文件 [{"id": ..., "user": ..., "assistant": ...}]

用法:
  python bootstrap_data.py -i raw_dialogs.json -o import.json
  python bootstrap_data.py -i raw_dialogs.csv -o import.json
  python bootstrap_data.py -i raw_dialogs.txt -o import.json --mode pairs
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


def load_json_or_jsonl(filepath: str) -> list:
    """加载 JSON 数组或 JSONL 文件。"""
    raw = Path(filepath).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return [data]
    except (json.JSONDecodeError, ValueError):
        pass

    lines = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return lines


def load_csv(filepath: str, user_col: str = "user", assistant_col: str = "assistant") -> list:
    """从 CSV 加载对话数据。"""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "user": row.get(user_col, "").strip(),
                "assistant": row.get(assistant_col, "").strip(),
            })
    return rows


def load_pairs_txt(filepath: str) -> list:
    """从纯文本文件加载对话对（每 2 行为一组）。"""
    lines = [l.strip() for l in Path(filepath).read_text(encoding="utf-8").split("\n")]
    lines = [l for l in lines if l]
    rows = []
    for i in range(0, len(lines) - 1, 2):
        rows.append({"user": lines[i], "assistant": lines[i + 1]})
    return rows


def convert_to_ls_format(rows: list, id_prefix: str = "conv") -> list:
    """转为 LabelStudio 导入格式。"""
    output = []
    for idx, row in enumerate(rows):
        user = row.get("user", "").strip()
        assistant = row.get("assistant", "").strip()
        if not user or not assistant:
            print(f"  跳过空消息 (行 {idx + 1})")
            continue
        output.append({
            "id": f"{id_prefix}{idx + 1:04d}",
            "user": user,
            "assistant": assistant,
        })
    return output


def main():
    parser = argparse.ArgumentParser(
        description="将原始对话数据转换为 LabelStudio 导入格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python bootstrap_data.py -i chats.json -o import.json
  python bootstrap_data.py -i chats.csv -o import.json
  python bootstrap_data.py -i chats.txt -o import.json --mode pairs
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="输入文件")
    parser.add_argument("-o", "--output", default="ls_import.json", help="输出文件名")
    parser.add_argument("--mode", choices=["auto", "csv", "pairs"], default="auto",
                        help="输入模式: auto(自动检测), csv, pairs(每2行一组)")
    parser.add_argument("--user-col", default="user", help="CSV 用户列名")
    parser.add_argument("--assistant-col", default="assistant", help="CSV 助手列名")
    parser.add_argument("--id-prefix", default="conv", help="ID 前缀")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    path = Path(args.input)
    ext = path.suffix.lower()

    # 检测模式
    if args.mode == "csv" or (args.mode == "auto" and ext == ".csv"):
        rows = load_csv(args.input, args.user_col, args.assistant_col)
    elif args.mode == "pairs":
        rows = load_pairs_txt(args.input)
    elif args.mode == "auto" and ext in (".json", ".jsonl"):
        rows = load_json_or_jsonl(args.input)
    else:
        rows = load_pairs_txt(args.input)

    if not rows:
        print("错误: 输入文件无有效数据", file=sys.stderr)
        sys.exit(1)

    output = convert_to_ls_format(rows, args.id_prefix)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  总输入: {len(rows)} 条")
    print(f"  有效数据: {len(output)} 条")
    print(f"  已输出: {args.output}")


if __name__ == "__main__":
    main()
