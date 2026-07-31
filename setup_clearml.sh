#!/bin/bash

# =========================================================================
# 🧪 ClearML 自動部署與設定腳本
# 職責：啟動 ClearML 本地端伺服器 (包含 Redis, Mongo, Elastic 等)
# =========================================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "======================================================="
echo "🧪 正在啟動 ClearML MLOps 後台管理平台..."
echo "======================================================="

# 啟動 ClearML 的 Docker 服務
echo "🐳 啟動 ClearML 相關 Docker 容器..."
docker-compose -f Fall/tools/docker-compose-clearml.yml up -d
sleep 3

echo "======================================================="
echo "✅ ClearML 伺服器已成功在背景啟動！"
echo "🌐 ClearML Web UI 戰情室: http://localhost:8080"
echo "🌐 API Server: http://localhost:8008"
echo "🌐 File Server: http://localhost:8081"
echo ""
echo "接下來的設定步驟："
echo "1. 打開瀏覽器進入 http://localhost:8080"
echo "2. 到右上角 Settings -> Workspace -> Create new credentials"
echo "3. 在終端機執行 'clearml-init' 並貼上剛剛產生的設定檔"
echo "======================================================="
