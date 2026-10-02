FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH="/app:/app/.."

# 安裝 uv 包管理器
RUN pip install --no-cache-dir uv

# 複製依賴描述檔並安裝套件（所有依賴均為預編譯 binary wheel，不需 gcc/libpq-dev）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev

# 複製後端原始碼
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
