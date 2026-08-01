#!/bin/bash

# =========================================================================
# 🏷️ Label Studio 與 YOLO ML 自動導入腳本
# 職責：啟動 Label Studio 容器，並啟動 Watchdog 負責 YOLO ML 自動預標註與導入
# =========================================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "======================================================="
echo "🏷️ 正在啟動 Label Studio 標註平台與自動導入系統..."
echo "======================================================="

# 1. 啟動 Label Studio 與 Webhook 監聽器
echo "🐳 [1/2] 啟動 Label Studio Docker 容器..."
docker-compose -f Fall/tools/docker-compose-label.yml up -d
sleep 3

# 2. 啟動 Watchdog (自動執行 inference_to_labelstudio_sdk.py 導入預標註資料)
echo "🔄 [2/2] 啟動 YOLO ML 自動導入程式 (Watchdog)..."
# 啟動前先砍掉舊的程序避免重複
pkill -f "python.*watchdog.py" >/dev/null 2>&1

PROJECT_PATH="$BASE_DIR/Fall"
TOOLS_PATH="$PROJECT_PATH/tools"

# 確保環境變數對齊
if [ -f "$TOOLS_PATH/.env" ]; then
    export $(grep -v '^#' "$TOOLS_PATH/.env" | xargs)
fi

nohup "$PROJECT_PATH/.venv/bin/python" "$TOOLS_PATH/watchdog.py" > "$BASE_DIR/watchdog_label_studio.log" 2>&1 &

echo "======================================================="
echo "✅ Label Studio 系統與自動導入腳本已啟動！"
echo "🌐 Label Studio 網頁: http://localhost:8082"
echo "💡 Watchdog 已在背景常駐，將會自動把 YOLO 辨識結果上傳到 Label Studio。"
echo "日誌可查看: watchdog_label_studio.log"
echo "======================================================="

# =========================================================================
# 📝 腳本備註：setup_label_studio.sh (標註專用)
# 功用：這是「日常開發與標註」專用的輕量級啟動開關。
# 只會單獨開啟 Label Studio 網頁，外加啟動 watchdog.py (自動畫骨架的巡邏員)。
# 當你有一批新照片要人工審核微調時開這個就好，不會佔用不必要的電腦資源。
# =========================================================================
