FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（rawpy需要libraw）
RUN apt-get update && apt-get install -y \
    libraw-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]