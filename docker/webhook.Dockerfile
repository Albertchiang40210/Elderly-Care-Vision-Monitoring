FROM python:3.13-slim

WORKDIR /app

# 依賴在 build 階段安裝，容器啟動時不再 pip install
RUN pip install --no-cache-dir fastapi uvicorn requests clearml boto3 opencv-python-headless numpy

CMD ["python", "Fall/tools/webhook_receiver.py"]
