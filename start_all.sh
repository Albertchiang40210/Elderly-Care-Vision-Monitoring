#!/bin/bash

# =========================================================================
# 🚀 終極一鍵啟動指令 (Start All Pipeline Master Script)
# =========================================================================

# 1. 確保腳本執行權限與路徑
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "======================================================="
echo "🔥 [一鍵點火] 正在開啟全連鎖安養中心 MLOps 智慧閉環系統..."
echo "======================================================="

# 2. 啟動第一步：Docker 基礎設施後台
echo "🐳 [1/3] 正在啟動所有 Docker 容器服務 (Triton, Kafka, ClearML, Label Studio)..."
./start_mlops_backend.sh

echo "⏳ 等待 Docker 服務暖身 10 秒..."
sleep 10

# 3. 啟動第二步：主動學習同步與監聽中樞
echo "🔄 [2/3] 正在啟動主動學習同步警衛與 Webhook 接收端..."
./start_full_auto.sh

# 4. 啟動第三步：最前線影像推論 Edge Worker
echo "🎬 [3/3] 正在點火最前線推理與微服務..."
if [ "$1" == "--headless" ]; then
    echo "🖥️  以 Headless 背景模式啟動前線影像辨識..."
    nohup ./start_inference.sh --headless > "$BASE_DIR/inference_system.log" 2>&1 &
    echo "======================================================="
    echo "🎉 [啟動完成] 所有服務已全數在背景順暢運行！"
    echo "💡 您可以關閉此終端機，並透過以下指令查看前線推論日誌："
    echo "   tail -f inference_system.log"
    echo "======================================================="
else
    echo "🖥️  以 GUI 視窗模式啟動前線影像辨識..."
    ./start_inference.sh
fi
