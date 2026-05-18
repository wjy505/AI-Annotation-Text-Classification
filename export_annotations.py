"""导出 LabelStudio 标注结果为 JSON 文件。

用法：
    python export_annotations.py --api-key <YOUR_TOKEN> [--project-id <ID>]

如果不指定 --project-id，将列出所有项目供选择。
"""

import json
import argparse
import sys

try:
    from label_studio_sdk import Client
except ImportError:
    print("请先安装 label-studio-sdk: pip install label-studio-sdk")
    sys.exit(1)


def list_projects(client):
    projects = client.list_projects()
    for p in projects:
        print(f"  ID: {p.id}  |  Title: {p.title}  |  Tasks: {p.task_number}")
    return projects


def export_project(client, project_id, output_file):
    project = client.get_project(project_id)
    annotations = project.export_tasks(export_type="JSON")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f"已导出 {len(annotations)} 条标注到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="导出 LabelStudio 标注结果")
    parser.add_argument("--api-key", required=True, help="LabelStudio Access Token")
    parser.add_argument("--url", default="http://localhost:8080", help="LabelStudio 地址")
    parser.add_argument("--project-id", type=int, help="项目 ID")
    parser.add_argument("--output", default="annotations_export.json", help="输出文件")
    args = parser.parse_args()

    client = Client(url=args.url, api_key=args.api_key)

    if args.project_id:
        export_project(client, args.project_id, args.output)
    else:
        print("未指定项目 ID，当前所有项目：")
        projects = list_projects(client)
        if not projects:
            print("没有找到任何项目。请先在 Web UI 中创建项目。")
            return
        try:
            pid = int(input("\n请输入要导出的项目 ID: "))
        except ValueError:
            print("无效的项目 ID")
            return
        export_project(client, pid, args.output)


if __name__ == "__main__":
    main()
