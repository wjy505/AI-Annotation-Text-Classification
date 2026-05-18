# Dockerfile — 对话质量分类 API 服务
# 构建: docker build -t dialog-classifier .
# 运行: docker run -p 8000:8000 dialog-classifier

FROM python:3.10-slim

LABEL maintainer="AI标注+文本分类项目"
LABEL description="对话质量有用/无用分类推理 API"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（国内镜像加速）
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers \
    fastapi \
    uvicorn \
    scikit-learn \
    numpy

# 复制代码和模型
COPY api_server.py .
COPY checkpoints/best_model/ ./checkpoints/best_model/

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
