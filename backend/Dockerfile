FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH="/app:/app/.."


# 安裝基本系統套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# 安裝 uv 包管理器
RUN pip install --no-cache-dir uv

# 複製依賴描述檔並安裝套件
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# 複製後端原始碼
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
