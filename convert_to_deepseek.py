"""
convert_to_deepseek.py — 将原始对话数据转换为 DeepSeek 微调 JSONL 格式。

功能:
  - 输入 JSON / CSV / LabelStudio 导出 JSON
  - 清洗空值、过短消息、首尾空白、重复换行
  - 只保留标注为"有用"的样本（可配置）
  - 可选过滤拒答模式（"抱歉"、"我不知道" 等）
  - 输出统计信息

用法:
  python convert_to_deepseek.py input.json -o train.jsonl
  python convert_to_deepseek.py input.csv -o train.jsonl --label-col quality --no-filter-refusal
  python convert_to_deepseek.py input.json -o train.jsonl --label-studio
"""

import json
import csv
import argparse
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Optional

# ─── 拒答关键词（语义一致 + 拒答直接相关） ───
DEFAULT_REFUSAL_PATTERNS = [
    "抱歉",
    "对不起",
    "我不知道",
    "我无法",
    "我不能",
    "恕我直言",
    "很抱歉",
    "无法回答",
    "没有足够的信息",
    "无法提供",
    "不在我的知识范围内",
    "作为一个AI",
    "作为AI",
]


def clean_text(text: str) -> str:
    """去除首尾空白，将 3 个以上连续换行压缩为两个换行。"""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_empty_or_short(text: str, min_len: int = 3) -> bool:
    """检查文本是否为空或过短（去除空白后不足 min_len 个字符）。"""
    return len(text.strip()) < min_len


def contains_refusal(text: str, patterns: list) -> bool:
    """检查文本是否包含拒答关键词。"""
    return any(p in text for p in patterns)


def flatten_label_studio(data: list) -> list:
    """解析 LabelStudio 导出的嵌套 JSON 为扁平列表。

    LabelStudio 导出格式:
      [{"id": ..., "data": {"user": "...", "assistant": "..."},
        "annotations": [{"result": [{"value": {"choices": [...]}}, ...]}]}]

    返回:
      [{"user": ..., "assistant": ..., "label": "useful"/"useless", "error_type": "..."}]
    """
    rows = []
    for task in data:
        row = {}
        # 取 data 字段中的原始文本
        task_data = task.get("data", {})
        row["user"] = task_data.get("user", "")
        row["assistant"] = task_data.get("assistant", "")

        annotations = task.get("annotations", [])
        if annotations:
            results = annotations[0].get("result", [])
            for r in results:
                rtype = r.get("type", "")
                value = r.get("value", {})
                if rtype == "choices":
                    choices = value.get("choices", [])
                    row["label"] = choices[0] if choices else ""
                elif rtype == "textarea":
                    row["error_type"] = value.get("text", [""])[0] if value.get("text") else ""
        rows.append(row)
    return rows


def load_json_or_jsonl(filepath: str) -> list:
    """自动识别 JSON / JSONL 文件，返回字典列表。"""
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8")

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
    if lines:
        return lines

    raise ValueError(f"无法解析输入文件: {filepath}")


def parse_input(filepath: str, label_studio: bool = False) -> list:
    """读取 JSON / JSONL / CSV 文件，返回字典列表。"""
    path = Path(filepath)
    if path.suffix.lower() == ".csv":
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    data = load_json_or_jsonl(filepath)

    if label_studio:
        return flatten_label_studio(data)

    # 自动检测 LabelStudio 格式：包含 data + annotations 字段
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and "data" in first and "annotations" in first:
            return flatten_label_studio(data)

    return data


def convert(
    input_path: str,
    output_path: str,
    label_col: str = "label",
    useful_value: str = "useful",
    user_col: str = "user",
    assistant_col: str = "assistant",
    filter_refusal: bool = True,
    refusal_patterns: Optional[list] = None,
    label_studio: bool = False,
    min_len: int = 3,
) -> dict:
    """主转换逻辑，返回统计信息。"""
    if refusal_patterns is None:
        refusal_patterns = DEFAULT_REFUSAL_PATTERNS

    rows = parse_input(input_path, label_studio)

    # 统一格式：将 messages 格式转为扁平 user/assistant
    normalized = []
    for row in rows:
        if "messages" in row:
            user = ""
            assistant = ""
            for m in row["messages"]:
                if m.get("role") == "user":
                    user += m.get("content", "") + "\n"
                elif m.get("role") == "assistant":
                    assistant += m.get("content", "") + "\n"
            normalized.append({
                "user": user.strip(),
                "assistant": assistant.strip(),
                "label": row.get("label", row.get("label_col", "")),
            })
        else:
            normalized.append(row)

    stats = Counter()
    stats["total_input"] = len(normalized)
    outputs = []

    for row in normalized:
        user = clean_text(row.get(user_col, "") if user_col in row else row.get("user", ""))
        assistant = clean_text(row.get(assistant_col, "") if assistant_col in row else row.get("assistant", ""))
        label = (row.get("label") or row.get(label_col, "")).strip().lower()

        # ── 过滤：标签非有用 ──
        if label and label != useful_value:
            stats["filtered_label_not_useful"] += 1
            continue

        # ── 过滤：user 为空或过短 ──
        if is_empty_or_short(user, min_len):
            stats["filtered_user_empty_or_short"] += 1
            continue

        # ── 过滤：assistant 为空或过短 ──
        if is_empty_or_short(assistant, min_len):
            stats["filtered_assistant_empty_or_short"] += 1
            continue

        # ── 过滤：拒答模式 ──
        if filter_refusal and contains_refusal(assistant, refusal_patterns):
            stats["filtered_refusal"] += 1
            continue

        outputs.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })

    stats["output_written"] = len(outputs)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return stats


def print_stats(stats: dict):
    """打印统计信息。"""
    print("\n" + "=" * 50)
    print("  转 换 统 计")
    print("=" * 50)
    print(f"  总输入行数:          {stats['total_input']:>6}")
    print(f"  保留行数:            {stats['output_written']:>6}")
    print("-" * 50)
    if stats.get("filtered_label_not_useful", 0):
        print(f"  过滤 - 标注非有用:   {stats['filtered_label_not_useful']:>6}")
    if stats.get("filtered_user_empty_or_short", 0):
        print(f"  过滤 - user 空/过短:  {stats['filtered_user_empty_or_short']:>6}")
    if stats.get("filtered_assistant_empty_or_short", 0):
        print(f"  过滤 - assistant 过短:{stats['filtered_assistant_empty_or_short']:>6}")
    if stats.get("filtered_refusal", 0):
        print(f"  过滤 - 拒答模式:      {stats['filtered_refusal']:>6}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="将对话数据转换为 DeepSeek 微调 JSONL 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_to_deepseek.py data.json -o train.jsonl
  python convert_to_deepseek.py data.csv -o train.jsonl --label-col quality
  python convert_to_deepseek.py export.json -o train.jsonl --label-studio
  python convert_to_deepseek.py data.json -o train.jsonl --no-filter-refusal
        """,
    )
    parser.add_argument("input", help="输入文件路径（JSON 或 CSV）")
    parser.add_argument("-o", "--output", default="train.jsonl", help="输出 JSONL 文件路径（默认 train.jsonl）")
    parser.add_argument("--label-col", default="label", help="标签列名（默认 label）")
    parser.add_argument("--useful-value", default="useful", help="标记为有用的值（默认 useful）")
    parser.add_argument("--user-col", default="user", help="用户消息列名（默认 user）")
    parser.add_argument("--assistant-col", default="assistant", help="助手回复列名（默认 assistant）")
    parser.add_argument("--no-filter-refusal", dest="filter_refusal", action="store_false",
                       help="关闭拒答过滤")
    parser.add_argument("--label-studio", action="store_true",
                       help="输入为 LabelStudio 导出格式，自动解析嵌套结构")
    parser.add_argument("--min-len", type=int, default=3, help="消息最短字符数（默认 3）")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    stats = convert(
        input_path=args.input,
        output_path=args.output,
        label_col=args.label_col,
        useful_value=args.useful_value,
        user_col=args.user_col,
        assistant_col=args.assistant_col,
        filter_refusal=args.filter_refusal,
        label_studio=args.label_studio,
        min_len=args.min_len,
    )

    print_stats(stats)
    print(f"已输出训练数据: {args.output}")


if __name__ == "__main__":
    main()
