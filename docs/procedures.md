# 完整操作步骤手册

本机环境：Windows 11，Python 3.10.10 (`D:/Python/pycharm/python`)

---

## 第一步：安装 VC++ 运行时（已装可跳过）

1. 浏览器打开 https://aka.ms/vs/17/release/vc_redist.x64.exe
2. 下载后双击安装，勾选"同意"，点击安装
3. 安装完成后 **重启终端**（关掉所有 CMD / VSCode 终端后重新打开）

---

## 第二步：安装 Python 依赖

> 用 PyCharm 自带的 Python，不要用 venv（venv 的 torch DLL 有兼容问题）。

打开 CMD 或 VSCode 终端，逐条执行：

```bash
# 1. 验证 Python 可用
D:/Python/pycharm/python --version
# 输出应为: Python 3.10.10

# 2. 安装 PyTorch（CPU 版）
D:/Python/pycharm/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 安装 HuggingFace + 训练依赖
D:/Python/pycharm/python -m pip install transformers accelerate scikit-learn numpy tqdm

# 4. 验证安装
D:/Python/pycharm/python -c "import torch; print('torch', torch.__version__)"
# 输出应为: torch 2.12.0+cpu  （没有报错即成功）
```

---

## 第三步：安装 LabelStudio（标注工具，单独安装）

```bash
# 创建独立的 venv（仅用于 label-studio，不需要 torch）
python -m venv ls_venv
ls_venv\Scripts\activate
pip install label-studio label-studio-sdk

# 启动服务
label-studio start
# → 浏览器打开 http://localhost:8080
# → 首次访问注册账号（邮箱 + 密码 + 确认密码）
```

> 之后每次启动：`ls_venv\Scripts\activate` → `label-studio start`

---

## 第四步：创建标注项目

1. 登录后点击蓝色 **Create** 按钮
2. **Project Name** 填 `对话文本标注`，点 **Save**
3. 进入项目 → 左侧点 **Settings**（齿轮图标）
4. 点 **Labeling Interface** → 切到 **Code** 标签
5. 清空默认内容，用记事本打开本项目 `labeling_config.xml`，复制全部内容粘贴进去
6. 点击 **Save** 保存

---

## 第五步：导入数据

1. 项目页面顶部点项目名称 **"对话文本标注"** 回到主页
2. 点击 **Import** 按钮
3. 选择本项目 `sample_conversations.json`，上传
4. 确认弹窗中字段显示正确，点导入

---

## 第六步：标注操作

1. 项目主页点击蓝色 **Label All Tasks** 按钮
2. 界面展示：
   - 左侧蓝色卡片 = 用户消息
   - 右侧绿色卡片 = 助手回复
   - 下方单选 = 有用 / 无用
   - 底部文本框 = 错误类型（选"无用"时填写）
3. 每条标注完后点右下角 **Submit**，自动跳下一条
4. 4 条全部标注完返回项目主页

---

## 第七步：导出标注结果

1. 项目页面 → 点击 **Export**
2. 选择 **JSON** 格式
3. 下载文件 → 放到项目目录下（`AI 标注 + 文本分类/`）
4. 改名为 `annotations.json`

---

## 第八步：数据清洗 + 质检

```bash
# 用系统 Python 执行
D:/Python/pycharm/python quality_check.py -i annotations.json -oc cleaned.jsonl -ox cls.jsonl --label-studio --balance-check
```

输出：
```
=======================================================
  质 检 报 告
=======================================================
  总样本数:            4
  通过:                4  (100.0%)
-------------------------------------------------------
  标签分布（分类数据集）:
    useful               2  ( 50.0%)
    useless              2  ( 50.0%)
=======================================================
  已输出清洗数据:   cleaned.jsonl  (4 条)
  已输出分类数据:   cls.jsonl  (4 条)
```

---

## 第九步：转换 DeepSeek 微调数据

```bash
D:/Python/pycharm/python convert_to_deepseek.py cleaned.jsonl -o train.jsonl
```

输出：
```
  总输入行数:               4
  保留行数:                 2
  过滤 - 标注非有用:        2
  已输出训练数据: train.jsonl
```

---

## 第十步：文本分类模型训练

> 国内网络必须设 HF 镜像，否则无法下载模型

```bash
# 设置镜像（当前终端有效）
set HF_ENDPOINT=https://hf-mirror.com

# 训练（CPU 模式）
D:/Python/pycharm/python train.py --data cls.jsonl --epochs 3 --batch-size 8 --no-cuda
```

输出关键行：
```
  设备: cpu
  数据加载: 4 条, 2 个类别: ['useful', 'useless']
  训练集: 3, 验证集: 1
  ...
  开始训练...
  {'eval_loss': '...', 'eval_accuracy': '...'}
  最佳模型已保存至: ./checkpoints/best_model
```

---

## 第十一步：模型推理

```bash
# 单条预测
D:/Python/pycharm/python predict.py -m checkpoints/best_model -t "用户: Python和Java区别？\n助手: Python动态类型，Java静态类型。"

# 批量预测（对训练数据全部推理一遍）
D:/Python/pycharm/python predict.py -m checkpoints/best_model -f cls.jsonl
```

输出示例：
```
  预测结果: useful  (置信度: 0.8521)

  Top-2 类别:
    1. useful        0.8521  █████████████████████████
    2. useless       0.1479  ████
```

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `WinError 1114` / DLL 失败 | torch 与 Python 不兼容 | 用 `D:/Python/pycharm/python`，不要用 venv 或 conda |
| `ConnectTimeout` / 连接超时 | huggingface.co 被墙 | `set HF_ENDPOINT=https://hf-mirror.com` |
| `No module named 'torch'` | 装错 Python 环境 | 确认用 `D:/Python/pycharm/python -m pip install torch` |
| `No module named 'accelerate'` | 缺包 | `D:/Python/pycharm/python -m pip install accelerate` |
| LabelStudio 打不开 | 服务没启动 | `ls_venv\Scripts\activate` → `label-studio start` |
| 导出 401 错误 | Token 类型不对 | 用 Access Token（不是 Refresh Token）；或直接用 Web UI Export |

---

## 正确 vs 错误用法速查

| 正确 | 错误 |
|------|------|
| `D:/Python/pycharm/python train.py ...` | `python train.py ...`（会调用 venv/conda 的损坏 Python） |
| `D:/Python/pycharm/python -m pip install ...` | `pip install ...`（会装到混乱的环境） |
| `ls_venv\Scripts\activate` → `label-studio start` | 用系统 Python 跑 label-studio（缺少 label-studio 包） |
| `set HF_ENDPOINT=...` 后再跑训练 | 不设 HF_ENDPOINT 直接训练（连不上 huggingface） |

---

## 第十二步：Docker 部署（可选）

### 安装 Docker Desktop

1. 官网下载：https://www.docker.com/products/docker-desktop/
2. 安装后重启系统
3. 验证：`docker --version`

### 构建镜像

```bash
cd "AI 标注 + 文本分类"
docker build -t dialog-classifier .
```

### 启动服务

```bash
# 仅 API
docker run -p 8000:8000 dialog-classifier

# 全栈（API + LabelStudio）
docker-compose up -d
```

### 调用

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"用户: Redis适合什么场景？\n助手: Redis适用于缓存、消息队列等场景。"}'
```

浏览器打开 `http://localhost:8000/docs` 使用 Swagger 交互式文档。
