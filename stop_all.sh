#!/bin/bash

# =========================================================================
# 🛑 全局緊急停止腳本 (Stop All Master Script)
# 職責：一鍵乾淨關閉前線 Edge 與後勤 Cloud 所有的 Docker 與微服務
# =========================================================================

echo "======================================================="
echo "🛑 正在強制關閉所有 AI 系統與微服務..."
echo "======================================================="

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. 關閉所有 Docker 容器 (Cloud)
echo "🐳 正在關閉 Docker 容器..."
(cd "$BASE_DIR" && docker-compose down >/dev/null 2>&1)
docker stop deepstream_pipeline >/dev/null 2>&1 || true
docker rm deepstream_pipeline >/dev/null 2>&1 || true

# 2. 關閉所有前線與後端微服務進程 (Edge & Cloud)
echo "🐍 正在強制終止 Python 與 Node 進程..."
pkill -f "python.*inference_test.py"
pkill -f "python.*vlm_worker.py"
pkill -f "python.*kafka_consumer.py"
pkill -f "python.*watchdog.py"
pkill -f "python.*webhook_receiver.py"
pkill -f "clearml-agent"
pkill -f "uvicorn.*main:app"
pkill -f "vite"
pkill -f "npm run dev"

# 3. 關閉串流與底層通訊
echo "📡 正在關閉串流與中繼服務..."
pkill -f "ffmpeg.*rtsp://localhost:8554"
pkill -f mediamtx
kill -9 $(lsof -t -i:9001) 2>/dev/null  # Webhook
kill -9 $(lsof -t -i:8000) 2>/dev/null  # FastAPI
kill -9 $(lsof -t -i:3000) 2>/dev/null  # React Frontend

echo "======================================================="
echo "✅ 所有系統進程皆已乾淨清理完畢！"
echo "======================================================="
