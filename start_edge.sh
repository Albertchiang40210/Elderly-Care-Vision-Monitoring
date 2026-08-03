#!/bin/bash

# =========================================================================
# 🛡️ 前線邊緣端駐守啟動腳本 (Edge Startup Script)
# 職責：24 小時緊盯畫面、推論、發送警報、攔截誤報
# =========================================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "======================================================="
echo "🛡️ [Edge 前線] 正在啟動 24H AI 推論與安防系統..."
echo "======================================================="

# 1. 啟動 FastAPI 後端與 Kafka 接收端
echo "⚙️ [1/4] 啟動後端 FastAPI 與警報接送端..."
pkill -f "python.*kafka_consumer.py" >/dev/null 2>&1
(cd "$BASE_DIR/backend" && nohup "$BASE_DIR/Fall/.venv/bin/python" kafka_consumer.py > "$BASE_DIR/kafka_consumer.log" 2>&1 &)

if ! curl -s http://localhost:8000/docs >/dev/null 2>&1; then
    pkill -f "uvicorn.*main:app" >/dev/null 2>&1
    (cd "$BASE_DIR/backend" && nohup "$BASE_DIR/Fall/.venv/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8000 > "$BASE_DIR/backend.log" 2>&1 &)
fi

# 2. 啟動前端 Web 介面
echo "💻 [2/4] 啟動前端 Web 即時戰情室..."
if ! curl -s http://localhost:3000 >/dev/null 2>&1 && ! curl -s http://localhost:5173 >/dev/null 2>&1; then
    pkill -f "vite" >/dev/null 2>&1
    (cd "$BASE_DIR/frontend" && nohup npm run dev > "$BASE_DIR/frontend.log" 2>&1 &)
fi
sleep 2

# 3. 啟動 MediaMTX 影像中繼站
echo "📡 [3/4] 啟動 MediaMTX 串流中繼伺服器..."
pkill -f mediamtx >/dev/null 2>&1
MEDIAMTX_CONF="$BASE_DIR/RTSP.MediaMTX_20260723/mediamtx(example).yml"
nohup mediamtx "$MEDIAMTX_CONF" > "$BASE_DIR/mediamtx.log" 2>&1 &
sleep 2

# 3.5 啟動 go2rtc (WebRTC 專用轉播)
echo "⚡ [3.5/4] 啟動 go2rtc (提供超低延遲 WebRTC)..."
pkill -f go2rtc >/dev/null 2>&1
if [ ! -f "$BASE_DIR/go2rtc" ]; then
    echo "   ⬇️ 正在下載 go2rtc (首次啟動需下載)..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        curl -L -s https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_mac_arm64.zip -o "$BASE_DIR/go2rtc.zip"
    else
        curl -L -s https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_mac_amd64.zip -o "$BASE_DIR/go2rtc.zip"
    fi
    unzip -q -o "$BASE_DIR/go2rtc.zip" -d "$BASE_DIR/"
    rm -f "$BASE_DIR/go2rtc.zip"
    chmod +x "$BASE_DIR/go2rtc"
fi
nohup "$BASE_DIR/go2rtc" -config "$BASE_DIR/go2rtc.yaml" > "$BASE_DIR/go2rtc.log" 2>&1 &
sleep 2

# =========================================================================
# 📸 攝影機來源設定
# 1. 真實手機/遠端 IP 攝影機 (填寫 RTSP 網址)
RTSP_URL=""
# 2. Iriun / 筆電內建鏡頭 / USB 視訊鏡頭 (0=關閉, 1=預設鏡頭, 2=外接鏡頭)
USE_WEBCAM="0"
# =========================================================================

if [ -n "$RTSP_URL" ]; then
    echo "📱 [遠端 IP Camera] 偵測到 RTSP 網址: $RTSP_URL"
    echo "📡 正在將實況影像推送至 rtsp://localhost:8554/cam_in"
    pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
    nohup ffmpeg -nostdin -rtsp_transport tcp -i "$RTSP_URL" -an -c:v copy -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
    sleep 2
elif [ "$USE_WEBCAM" != "0" ]; then
    CAM_INDEX=$((USE_WEBCAM - 1))
    echo "📷 [實體/虛擬鏡頭] 啟用 Mac 視訊鏡頭 (Index: $CAM_INDEX)！"
    echo "📡 正在將實體鏡頭畫面推送至 rtsp://localhost:8554/cam_in，以供應前端高清背景..."
    pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
    # 解決 Mac Iriun 攝影機的 avfoundation 影格率 bug (強制指定 60fps)
    # 搭配智慧滿版裁切 (Crop-to-fill)，強制消除手機直向帶來的巨大黑邊
    nohup ffmpeg -nostdin -f avfoundation -pixel_format uyvy422 -framerate 60 -i "$CAM_INDEX" -vf "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_in > /dev/null 2>&1 &
    sleep 3
else
    # Stage Demo 版：強制播放 4 個測試影片供四宮格使用
    TEST_VIDEO_1="$BASE_DIR/Fall/test_demo/test1.mp4"
    TEST_VIDEO_2="$BASE_DIR/Fall/test_demo/test2.mp4"
    TEST_VIDEO_3="$BASE_DIR/Fall/test_demo/test3.mp4"
    TEST_VIDEO_4="$BASE_DIR/Fall/test_demo/test4.mp4"
    
    # 暫時以 test1.mp4 作為預設影片，避免檔案不存在導致 ffmpeg 崩潰
    [ ! -f "$TEST_VIDEO_1" ] && TEST_VIDEO_1="$BASE_DIR/Fall/test_demo/test1.mp4"
    [ ! -f "$TEST_VIDEO_2" ] && TEST_VIDEO_2="$BASE_DIR/Fall/test_demo/test1.mp4"
    [ ! -f "$TEST_VIDEO_3" ] && TEST_VIDEO_3="$BASE_DIR/Fall/test_demo/test1.mp4"
    [ ! -f "$TEST_VIDEO_4" ] && TEST_VIDEO_4="$BASE_DIR/Fall/test_demo/test1.mp4"

    if [ -f "$TEST_VIDEO_1" ]; then
        echo "🎥 [展示模式] 未設定攝影機，啟動四宮格多路推流 (Demo Mode)"
        pkill -f "ffmpeg.*rtsp://localhost:8554" >/dev/null 2>&1
        nohup ffmpeg -nostdin -re -i "$TEST_VIDEO_1" -vf "scale=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_0 > /dev/null 2>&1 &
        nohup ffmpeg -nostdin -re -i "$TEST_VIDEO_2" -vf "scale=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_1 > /dev/null 2>&1 &
        nohup ffmpeg -nostdin -re -i "$TEST_VIDEO_3" -vf "scale=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_2 > /dev/null 2>&1 &
        nohup ffmpeg -nostdin -re -i "$TEST_VIDEO_4" -vf "scale=1280:720" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -r 30 -f rtsp rtsp://localhost:8554/cam_3 > /dev/null 2>&1 &
        sleep 2
    fi
fi

# 4. 啟動 VLM 與最前線推論引擎
echo "🧠 [4/4] 啟動 VLM 警報攔截器與核心 AI 辨識引擎..."
pkill -f "python.*vlm_worker.py" >/dev/null 2>&1
(cd "$BASE_DIR/Fall/tools" && nohup "$BASE_DIR/Fall/.venv/bin/python" vlm_worker.py > "$BASE_DIR/vlm_worker.log" 2>&1 &)

echo "======================================================="
echo "✅ [Edge 前線] 系統啟動完畢，AI 已進入 24H 監視狀態！"
echo "🔗 請在瀏覽器開啟戰情室: http://localhost:5173"
echo "======================================================="

# 硬體自動偵測：若有 N 卡則啟用 DeepStream Docker，否則使用 Python (GStreamer+OpenCV)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🚀 [硬體偵測] 發現 NVIDIA GPU，自動切換至 DeepStream 產線級引擎！"
    
    # 清理舊的容器
    docker stop deepstream_pipeline >/dev/null 2>&1 || true
    docker rm deepstream_pipeline >/dev/null 2>&1 || true
    
    # 啟怒 DeepStream 容器 (請根據您實際的 image 名稱與需求微調以下參數)
    docker run -d --name deepstream_pipeline --gpus all \
        -v "$BASE_DIR/deepstream_configs:/app/deepstream_configs" \
        -w /app \
        nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
        deepstream-app -c deepstream_configs/deepstream_app_config.txt
        
    echo "✅ DeepStream 已在背景啟動 (容器: deepstream_pipeline)"
else
    echo "💻 [硬體偵測] 未發現 NVIDIA GPU，自動降級啟用 Python (GStreamer+OpenCV) 引擎！"
    (cd "$BASE_DIR/Fall/tools" && "$BASE_DIR/Fall/.venv/bin/python" inference_test.py --headless)
fi

# =========================================================================
# 📝 腳本備註：start_edge.sh (前線士兵)
# 功用：這是模擬裝在病房攝影機裡的邊緣運算程式。
# 它會開啟攝影機畫面、開始跑即時的 AI 推論、開啟護理站的警告網頁，
# 並在有人跌倒時發送警報給後台。
# 適合與 start_cloud.sh 搭配，做完整的落地實戰 Demo。
# =========================================================================
