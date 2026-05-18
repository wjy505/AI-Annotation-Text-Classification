"""batch_prelabel.py — 批量预标注脚本。

用法：
  先获取 Token：浏览器 http://localhost:8080 → 右上角头像 → Account & Settings → Access Token
  python batch_prelabel.py --token <你的TOKEN> [--project-id <ID>]
"""

import argparse, json, os, sys, requests, torch, numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def get_projects(headers, base):
    r = requests.get(f"{base}/api/projects", headers=headers)
    return r.json() if isinstance(r.json(), list) else r.json().get("results", [])

def get_tasks(headers, base, project_id):
    """获取项目所有任务。"""
    tasks = []
    page = 1
    while True:
        r = requests.get(
            f"{base}/api/tasks",
            headers=headers,
            params={"project": project_id, "page": page, "page_size": 100},
        )
        data = r.json()
        items = data if isinstance(data, list) else data.get("results", [])
        if not items:
            break
        tasks.extend(items)
        if len(items) < 100:
            break
        page += 1
    return tasks

def main():
    parser = argparse.ArgumentParser(description="批量预标注 LabelStudio 任务")
    parser.add_argument("--token", required=True, help="LabelStudio Access Token")
    parser.add_argument("--project-id", type=int, help="项目 ID（不指定则选最新）")
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--model", default="checkpoints/best_model")
    args = parser.parse_args()

    headers = {"Authorization": f"Token {args.token}", "Content-Type": "application/json"}
    base = args.base

    # 1. 选项目
    projects = get_projects(headers, base)
    if not projects:
        print("没有找到项目（Token 可能无效）")
        sys.exit(1)

    if args.project_id:
        project = next((p for p in projects if p["id"] == args.project_id), None)
    else:
        print("所有项目：")
        for p in projects:
            print(f"  ID={p['id']}  |  {p.get('title','')}  |  {p.get('task_number',0)} 条")
        project = projects[-1]  # 最新

    if not project:
        print("项目不存在")
        sys.exit(1)

    print(f"\n选中项目: {project['title']}")

    # 2. 获取任务
    tasks = get_tasks(headers, base, project["id"])
    print(f"任务总数: {len(tasks)}")

    # 3. 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with open(os.path.join(args.model, "label_map.json"), encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    # 4. 逐条预测并写入
    created, skipped = 0, 0
    for task in tasks:
        if task.get("total_predictions", 0) > 0:
            skipped += 1
            continue

        user = task.get("data", {}).get("user", "")
        assistant = task.get("data", {}).get("assistant", "")
        if not user or not assistant:
            continue

        text = f"用户: {user}\n助手: {assistant}"
        enc = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=256, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        label = id2label[top_idx]
        confidence = float(probs[top_idx])

        pred = {
            "model_version": "1.0",
            "result": [{
                "from_name": "usefulness",
                "to_name": "user_msg",
                "type": "choices",
                "value": {"choices": [label]},
            }],
            "score": confidence,
        }

        r = requests.post(
            f"{base}/api/tasks/{task['id']}/predictions",
            headers=headers, json=pred,
        )
        if r.status_code == 201:
            created += 1
            print(f"  ✓ task {task['id']:>4d}: {label:<8s} ({confidence:.3f})")
        else:
            print(f"  ✗ task {task['id']:>4d}: {r.status_code} {r.text[:60]}")

    print(f"\n完成: {created} 条新预测, 跳过 {skipped} 条（已有预测）")

if __name__ == "__main__":
    main()
