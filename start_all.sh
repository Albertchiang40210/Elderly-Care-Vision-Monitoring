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

# 2.5 自動在背景啟動 MediaMTX 串流中繼伺服器 (支援業界真實 IP Camera 與本地模擬)
echo "📡 [MediaMTX & RTSP] 正在啟動 MediaMTX 生產級串流伺服器與中繼轉接..."
pkill -f mediamtx >/dev/null 2>&1
MEDIAMTX_CONF="$BASE_DIR/RTSP.MediaMTX_20260723/mediamtx(example).yml"
nohup mediamtx "$MEDIAMTX_CONF" > "$BASE_DIR/mediamtx.log" 2>&1 &
sleep 2

TEST_VIDEO="$BASE_DIR/Fall/test_demo/test1.mp4"
USE_CAMERA=0
CAM_NAME=""

# 🌟 業界生產模式 (Industry Production Mode)：若有設定真實 IP Camera RTSP 網址
if [ -n "$RTSP_CAMERA_URL" ]; then
    echo "🏢 [業界生產模式] 偵測到真實 IP Camera RTSP 網址: $RTSP_CAMERA_URL"
    echo "📡 正在將 MediaMTX 直連 IP Camera 並轉碼轉播至 cam_in 頻道..."
    pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
    nohup ffmpeg -nostdin -rtsp_transport tcp -i "$RTSP_CAMERA_URL" -an -c:v copy -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
    sleep 2
else
    # 💻 開發與測試模式 (Dev & POC Mode)：本地相機 / 測試影片自動推流
    # 1. 優先測試 Iriun Camera (Pixel 手機)
    if ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -q "Iriun Camera"; then
        echo "📱 偵測到 Iriun Camera 裝置，正在測試手機連線狀態..."
        if ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 60 -i "Iriun Camera" -t 1 -f null - >/dev/null 2>&1; then
            USE_CAMERA=1
            CAM_NAME="Iriun Camera"
        else
            echo "⚠️ Iriun Camera 尚未實態傳輸（請確認 Pixel 手機上的 Iriun App 已開啟）。"
        fi
    fi

    # 2. 備選 Mac 內建視訊相機 (Device 0)
    if [ "$USE_CAMERA" -eq 0 ]; then
        if ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 30 -i "0" -t 1 -f null - >/dev/null 2>&1; then
            echo "💻 檢測到 Mac 內建視訊鏡頭可用，自動切換至本機攝影機串流..."
            USE_CAMERA=1
            CAM_NAME="0"
        fi
    fi

    if [ "$USE_CAMERA" -eq 1 ]; then
        echo "📱 [Live Camera] 相機實時連線成功 ($CAM_NAME)！開啟極低延遲 16:9 推流..."
        pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
        nohup ffmpeg -nostdin -f avfoundation -pixel_format uyvy422 -framerate 60 -i "$CAM_NAME" -vf "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
        sleep 2
    elif [ -f "$TEST_VIDEO" ]; then
        echo "🎥 [DEMO 模式] 降級使用預設影片 RTSP 推流: $TEST_VIDEO -> rtsp://localhost:8554/cam_in"
        pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
        nohup ffmpeg -nostdin -re -stream_loop -1 -i "$TEST_VIDEO" -an -c:v copy -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
        sleep 2
    fi
fi

# 3. 啟動第二步：主動學習同步與監聽中樞
echo "🔄 [2/3] 正在啟動主動學習同步警衛與 Webhook 接收端..."
./start_full_auto.sh

# 4. 啟動第三步：最前線影像推論 Edge Worker (智慧自動偵測 N 卡 / 雙模式)
HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA_GPU=1
elif command -v docker >/dev/null 2>&1 && docker info 2>&1 | grep -iq "nvidia"; then
    HAS_NVIDIA_GPU=1
fi

# 若使用者未手動指定 ENGINE_MODE，則依據硬體自動決策
if [ -z "$ENGINE_MODE" ]; then
    if [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
        ENGINE_MODE="prod"
        echo "🔍 [硬體檢測] 成功偵測到 NVIDIA GPU 顯卡！系統自動選擇 Prod 模式 (DeepStream)..."
    else
        ENGINE_MODE="dev"
        echo "🔍 [硬體檢測] 未偵測到 NVIDIA GPU 顯卡 (Mac/CPU 環境)，系統自動選擇 Dev 模式 (Python)..."
    fi
else
    echo "⚙️ [手動模式] 已指定 ENGINE_MODE=$ENGINE_MODE"
fi

echo "🎬 [3/3] 正在點火最前線推理與微服務 (模式: $ENGINE_MODE)..."
echo "======================================================="
echo "🎉 [點火完成] 所有服務已全數在終端機啟動並即時印出日誌！"
echo "💡 提示：按 [Ctrl + C] 可隨時一鍵自動停止所有系統服務。"
echo "======================================================="

# 設定按 Ctrl+C 時自動一鍵關閉清理所有推流與微服務
cleanup() {
    echo -e "\n🛑 偵測到中斷訊號 (Ctrl+C)，正在一鍵自動清理所有推流與微服務..."
    docker stop deepstream_pipeline >/dev/null 2>&1 || true
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

# 動態依據 ENGINE_MODE 選擇推理引擎
if [ "$ENGINE_MODE" = "prod" ]; then
    echo "🚀 [Prod 模式] 啟動 NVIDIA DeepStream 高效能 GPU Pipeline..."
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        docker run --gpus all --rm --net=host --name deepstream_pipeline \
            -v "$BASE_DIR/deepstream_configs:/configs" \
            nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
            deepstream-app -c /configs/deepstream_app_config.txt
    else
        echo "⚠️ 偵測到 Docker 無 GPU 執行權限，安全自動降級至 Dev 模式 (Python + OpenCV)..."
        (cd "$BASE_DIR/Fall/tools" && "$BASE_DIR/Fall/.venv/bin/python" inference_test.py --headless)
    fi
else
    echo "💻 [Dev 模式] 啟動 Mac 本地 Python 視覺推論引擎 (OpenCV + ONNX/Triton)..."
    (cd "$BASE_DIR/Fall/tools" && "$BASE_DIR/Fall/.venv/bin/python" inference_test.py --headless)
fi

# =========================================================================
# 💡 [檔案說明與核心職責]
# 「它是本專案的『一鍵總點火開關 (Master Master Command Script)』。」
# 執行本腳本將依序點連鎖啟動：
# 1. start_mlops_backend.sh：啟動 Docker 基礎設施 (Kafka, ClearML, Label Studio, PostgreSQL)
# 2. backend/kafka_consumer.py：啟動訊息轉接站將告警寫入資料庫
# 3. MediaMTX & FFmpeg：啟動 RTSP 生產級影音串流中繼伺服器與推流
# 4. start_full_auto.sh：啟動 VLM 護理長大腦 (Qwen2.5-VL)、Webhook 點火閥與 Watchdog
# 5. Fall/tools/inference_test.py：前台引爆 OpenCV 多鏡頭邊緣 AI 巡邏推論引擎
# 按下 [Ctrl + C] 會自動觸發 cleanup() 清理釋放所有背景進程與串流頻道。
# =========================================================================

