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

# 2.2 確定 FastAPI 後端在 Docker 容器內運行 (nh-backend) 並啟動 Kafka 轉接站
echo "⚙️ [FastAPI & Consumer] 啟動 Kafka 訊息轉接器 (processed-reports -> Postgres DB)..."
pkill -f "python.*kafka_consumer.py" >/dev/null 2>&1
(cd "$BASE_DIR/backend" && nohup "$BASE_DIR/Fall/.venv/bin/python" kafka_consumer.py > "$BASE_DIR/kafka_consumer.log" 2>&1 &)
sleep 2

# 2.5 自動在背景啟動 MediaMTX 與 FFmpeg RTSP 模擬推流
echo "📡 [MediaMTX & RTSP] 正在自動啟動 MediaMTX 串流伺服器與影片推流..."
pkill -f mediamtx >/dev/null 2>&1
MEDIAMTX_CONF="$BASE_DIR/RTSP.MediaMTX_20260723/mediamtx(example).yml"
nohup mediamtx "$MEDIAMTX_CONF" > "$BASE_DIR/mediamtx.log" 2>&1 &
sleep 2

TEST_VIDEO="$BASE_DIR/Fall/test_demo/test1.mp4"
USE_CAMERA=0
CAM_NAME=""

# 1. 優先測試 Iriun Camera (Pixel 手機，需要 framerate 60 模式)
if ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -q "Iriun Camera"; then
    echo "📱 偵測到 Iriun Camera 裝置，正在測試手機連線狀態..."
    if ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 60 -i "Iriun Camera" -t 1 -f null - >/dev/null 2>&1; then
        USE_CAMERA=1
        CAM_NAME="Iriun Camera"
    else
        echo "⚠️ Iriun Camera 尚未實態傳輸（請確認 Pixel 手機上的 Iriun App 已開啟並在 Mac 端 Iriun 視窗看到畫面）。"
    fi
fi

# 2. 若手機未連線，嘗試備選 Mac 內建視訊相機 (Device 0)
if [ "$USE_CAMERA" -eq 0 ]; then
    if ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 30 -i "0" -t 1 -f null - >/dev/null 2>&1; then
        echo "💻 檢測到 Mac 內建視訊鏡頭可用，自動切換至本機攝影機串流..."
        USE_CAMERA=1
        CAM_NAME="0"
    fi
fi

if [ "$USE_CAMERA" -eq 1 ]; then
    echo "📱 [Live Camera] 相機實時連線成功 ($CAM_NAME)！正在自動掛載極低延遲 16:9 比例等比防變形 RTSP 推流..."
    pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
    nohup ffmpeg -nostdin -f avfoundation -pixel_format uyvy422 -framerate 60 -i "$CAM_NAME" -vf "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
    sleep 2
elif [ -f "$TEST_VIDEO" ]; then
    echo "🎥 降級使用預設影片 RTSP 推流: $TEST_VIDEO -> rtsp://localhost:8554/cam_in"
    pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
    nohup ffmpeg -nostdin -re -stream_loop -1 -i "$TEST_VIDEO" -an -c:v copy -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
    sleep 2
fi

# 3. 啟動第二步：主動學習同步與監聽中樞
echo "🔄 [2/3] 正在啟動主動學習同步警衛與 Webhook 接收端..."
./start_full_auto.sh

# 4. 啟動第三步：最前線影像推論 Edge Worker
echo "🎬 [3/3] 正在點火最前線推理與微服務 (前台即時輸出模式)..."
echo "======================================================="
echo "🎉 [點火完成] 所有服務已全數在終端機啟動並即時印出日誌！"
echo "💡 提示：按 [Ctrl + C] 可隨時一鍵自動停止所有系統服務。"
echo "======================================================="

# 設定按 Ctrl+C 時自動一鍵關閉清理所有推流與微服務
cleanup() {
    echo -e "\n🛑 偵測到中斷訊號 (Ctrl+C)，正在一鍵自動清理所有推流與微服務..."
    pkill -f "python.*inference_test.py" >/dev/null 2>&1
    pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
    pkill -f mediamtx >/dev/null 2>&1
    pkill -f "python.*webhook_receiver.py" >/dev/null 2>&1
    pkill -f "python.*s3_sync_worker.py" >/dev/null 2>&1
    pkill -f "python.*vlm_worker.py" >/dev/null 2>&1
    pkill -f "python.*deployment_agent.py" >/dev/null 2>&1
    echo "✅ 所有推流與 AI 微服務已全數清理完畢！"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 前台即時印出推論與 AI 運作 Log
(cd "$BASE_DIR/Fall/tools" && "$BASE_DIR/Fall/.venv/bin/python" inference_test.py --headless)
