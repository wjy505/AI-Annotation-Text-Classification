"""
quality_check.py — 质检 + 自动修复脚本。

读取原始标注数据，清洗后输出两个文件:
  - cleaned_data.jsonl        → DeepSeek 微调用（messages 格式）
  - classification_data.jsonl → 文本分类训练用（text + label）

用法:
  python quality_check.py -i raw_data.json -oc cleaned.jsonl -ox cls.jsonl
  python quality_check.py -i export.json -oc cleaned.jsonl -ox cls.jsonl --label-studio
  python quality_check.py -i data.jsonl -oc cleaned.jsonl -ox cls.jsonl --balance-check
  python quality_check.py -i raw.jsonl -oc cleaned.jsonl -ox cls.jsonl --no-dedup
"""

import json
import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# ════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════
VALID_ROLES = {"user", "assistant", "system", "tool"}
CORE_ROLES = {"user", "assistant"}

ERROR_LABELS = {
    "messages_empty_or_not_list": "messages 为空/非列表",
    "message_not_dict":           "message 元素非字典",
    "invalid_role":               "非法 role 值",
    "empty_content":              "空 content",
    "all_empty_after_repair":     "修复后无可保留消息",
    "duplicate":                  "重复样本（MD5）",
}


# ════════════════════════════════════════════════
#  基础工具
# ════════════════════════════════════════════════

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


def flatten_label_studio(data: list) -> list:
    """LabelStudio 嵌套导出 → 扁平列表 [{user, assistant, label, error_type}]。"""
    rows = []
    for task in data:
        row = {"user": "", "assistant": "", "label": "", "error_type": ""}
        td = task.get("data", {})
        row["user"] = td.get("user", "")
        row["assistant"] = td.get("assistant", "")

        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                v = r.get("value", {})
                if r.get("type") == "choices":
                    choices = v.get("choices", [])
                    row["label"] = choices[0] if choices else ""
                elif r.get("type") == "textarea":
                    row["error_type"] = (v.get("text", [""]) or [""])[0]
        rows.append(row)
    return rows


def parse_input(filepath: str, label_studio: bool = False) -> list:
    """统一输入解析入口。"""
    data = load_json_or_jsonl(filepath)
    if not data:
        return []

    # 已是 DeepSeek messages 格式
    if isinstance(data[0], dict) and "messages" in data[0]:
        return data

    if label_studio:
        return flatten_label_studio(data)

    # 自动检测 LabelStudio 嵌套格式
    if isinstance(data[0], dict) and "data" in data[0] and "annotations" in data[0]:
        return flatten_label_studio(data)

    return data


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def fix_unicode_escapes(text: str) -> str:
    """修复字面量 \\uXXXX 为中文字符。"""
    if not text:
        return text

    def replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except (ValueError, OverflowError):
            return match.group(0)

    return re.sub(r"\\u([0-9a-fA-F]{4})", replace_unicode, text)


def normalize_text(text: str) -> str:
    """标准化文本：修复 Unicode、去首尾空白、压缩多余换行。"""
    text = fix_unicode_escapes(text)
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ════════════════════════════════════════════════
#  校验 & 修复
# ════════════════════════════════════════════════

def check_messages(msgs: list) -> tuple:
    """校验 messages 列表。返回 (ok: bool, error_key: str)。"""
    if not isinstance(msgs, list) or len(msgs) == 0:
        return False, "messages_empty_or_not_list"
    for m in msgs:
        if not isinstance(m, dict):
            return False, "message_not_dict"
        if m.get("role", "") not in VALID_ROLES:
            return False, "invalid_role"
        content = m.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return False, "empty_content"
    return True, ""


def repair_messages(msgs: list) -> Optional[list]:
    """修复：剔除空 content、非 CORE_ROLES、标准化文本。"""
    repaired = []
    for m in msgs:
        role = m.get("role", "")
        if role not in CORE_ROLES:
            continue
        content = normalize_text(m.get("content", ""))
        if not content:
            continue
        repaired.append({"role": role, "content": content})
    return repaired if repaired else None


def extract_dialogue(msgs: list) -> tuple:
    """提取 user 和 assistant 文本（用于去重签名）。"""
    u_parts, a_parts = [], []
    for m in msgs:
        if m["role"] == "user":
            u_parts.append(m["content"])
        elif m["role"] == "assistant":
            a_parts.append(m["content"])
    return "\n".join(u_parts), "\n".join(a_parts)


def build_classification(user: str, assistant: str, label: str) -> dict:
    return {"text": f"用户: {user}\n助手: {assistant}", "label": label}


def process_samples(samples: list, skip_dedup: bool = False) -> tuple:
    """主处理逻辑。

    Args:
      samples:   样本列表（messages 格式或扁平 user/assistant/label 格式）
      skip_dedup: True 时跳过去重

    Returns:
      clean:    清洗后的 messages 列表 [{"messages": [...]}]
      cls_data: 分类列表 [{"text": "...", "label": "..."}]
      report:   Counter 统计
    """
    clean, cls_data = [], []
    seen = set()
    report = Counter()
    report["total_input"] = len(samples)

    for sample in samples:
        # ── messages 格式 ──
        if "messages" in sample:
            msgs = sample["messages"]
            label = sample.get("label", "")
            ok, err_key = check_messages(msgs)
            if not ok:
                report[err_key] += 1
                continue

            repaired = repair_messages(msgs)
            if repaired is None:
                report["all_empty_after_repair"] += 1
                continue

            user_text, assistant_text = extract_dialogue(repaired)

            if not skip_dedup:
                key = md5_hash(user_text + "\x00" + assistant_text)
                if key in seen:
                    report["duplicate"] += 1
                    continue
                seen.add(key)

            report["passed"] += 1
            clean.append({"messages": repaired, "label": label})
            cls_data.append(build_classification(user_text, assistant_text, label))
            continue

        # ── 扁平 user/assistant/label 格式 ──
        user = normalize_text(sample.get("user", ""))
        assistant = normalize_text(sample.get("assistant", ""))
        label = sample.get("label", "")

        if not user or not assistant:
            report["empty_content"] += 1
            continue

        if not skip_dedup:
            key = md5_hash(user + "\x00" + assistant)
            if key in seen:
                report["duplicate"] += 1
                continue
            seen.add(key)

        report["passed"] += 1
        clean.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "label": label,
        })
        cls_data.append(build_classification(user, assistant, label))

    return clean, cls_data, report


# ════════════════════════════════════════════════
#  输出
# ════════════════════════════════════════════════

def write_jsonl(filepath: str, data: list):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_report(report: Counter, cls_data: list = None, balance_check: bool = False):
    total = report["total_input"]
    passed = report.get("passed", 0)
    failed = total - passed

    print("\n" + "=" * 55)
    print("  质 检 报 告")
    print("=" * 55)
    print(f"  总样本数:       {total:>6}")
    print(f"  通过:           {passed:>6}  ({passed / max(total, 1) * 100:.1f}%)")
    print(f"  失败:           {failed:>6}  ({failed / max(total, 1) * 100:.1f}%)")
    print("-" * 55)

    shown = False
    for key, desc in ERROR_LABELS.items():
        count = report.get(key, 0)
        if count > 0:
            shown = True
            print(f"  {desc:<24s} {count:>6}")
    if not shown:
        print("  (无已知错误类型)")

    if balance_check and cls_data:
        print("-" * 55)
        label_counts = Counter(d.get("label", "(无标签)") for d in cls_data)
        total_cls = len(cls_data)
        print("  标签分布（分类数据集）:")
        for label, count in label_counts.most_common():
            pct = count / max(total_cls, 1) * 100
            bar = "█" * int(pct / 5)
            print(f"    {label:<16s} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("=" * 55 + "\n")


# ════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="质检 + 自动修复标注数据 → DeepSeek 微调 & 文本分类 JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quality_check.py -i raw_data.json -oc cleaned.jsonl -ox cls.jsonl
  python quality_check.py -i export.json -oc cleaned.jsonl -ox cls.jsonl --label-studio
  python quality_check.py -i data.jsonl -oc cleaned.jsonl -ox cls.jsonl --balance-check
  python quality_check.py -i data.jsonl -oc cleaned.jsonl -ox cls.jsonl --no-dedup
        """,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="输入文件（JSON / JSONL / LabelStudio 导出）")
    parser.add_argument("-oc", "--output-clean", default="cleaned_data.jsonl",
                        help="清洗后微调 JSONL（默认 cleaned_data.jsonl）")
    parser.add_argument("-ox", "--output-class", default="classification_data.jsonl",
                        help="文本分类 JSONL（默认 classification_data.jsonl）")
    parser.add_argument("--label-studio", action="store_true",
                        help="输入为 LabelStudio 导出格式")
    parser.add_argument("--balance-check", action="store_true",
                        help="输出分类标签平衡性检查")
    parser.add_argument("--no-dedup", action="store_true",
                        help="关闭 MD5 去重")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    samples = parse_input(args.input, args.label_studio)
    clean, cls_data, report = process_samples(samples, skip_dedup=args.no_dedup)

    write_jsonl(args.output_clean, clean)
    write_jsonl(args.output_class, cls_data)

    print_report(report, cls_data, args.balance_check)
    print(f"  已输出清洗数据:   {args.output_clean}  ({len(clean)} 条)")
    print(f"  已输出分类数据:   {args.output_class}  ({len(cls_data)} 条)")


if __name__ == "__main__":
    main()
