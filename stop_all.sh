#!/bin/bash

# =========================================================================
# 🛑 一鍵安全關閉系統 (Stop All Pipeline Script)
# =========================================================================

echo "======================================================="
echo "🛑 正在一鍵安全關閉全連鎖安養中心 MLOps 系統..."
echo "======================================================="

# 1. 關閉前線推論與串流進程
echo "🎬 關閉前線影像推論與 FFmpeg 推流..."
pkill -f "inference_test.py" >/dev/null 2>&1
pkill -f "ffmpeg.*rtsp" >/dev/null 2>&1
pkill -f "mediamtx" >/dev/null 2>&1

# 2. 關閉 MLOps 監控與背景 Worker
echo "🔄 關閉主動學習監控、Webhook 監聽器與 VLM 大腦..."
pkill -f "watchdog.py" >/dev/null 2>&1
pkill -f "webhook_receiver.py" >/dev/null 2>&1
pkill -f "vlm_worker.py" >/dev/null 2>&1
pkill -f "model_deployment_agent.py" >/dev/null 2>&1
pkill -f "clearml-agent" >/dev/null 2>&1

# 3. 關閉後端服務
echo "⚙️ 關閉 FastAPI 後端與 Kafka Consumer..."
pkill -f "uvicorn.*main:app" >/dev/null 2>&1
pkill -f "kafka_consumer.py" >/dev/null 2>&1

echo "======================================================="
echo "✅ 所有背景影片推流與 AI 微服務已全數安全停止！"
echo "======================================================="

# =========================================================================
# 💡 [檔案說明與核心職責]
# 「它是本專案的『一鍵緊急煞車與資源清理腳本 (System Shutdown Master Script)』。」
# 執行本腳本會精準掃描並釋放所有運作中的進程與連接埠：
# 1. 關閉前線影像推論 (inference_test.py) 與 FFmpeg / MediaMTX 串流服務
# 2. 關閉主動學習組件 (watchdog.py, webhook_receiver.py, vlm_worker.py, clearml-agent)
# 3. 關閉後端微服務 (FastAPI/Uvicorn, kafka_consumer.py)
# 確保無殘留背景進程占用 GPU / CPU 資源或佔用 8000/8089/9001/8554 等 Port 號。
# =========================================================================

