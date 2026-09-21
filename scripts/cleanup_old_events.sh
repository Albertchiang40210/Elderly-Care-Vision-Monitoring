#!/bin/bash
# 🧹 清理超過 7 天的舊事件影片與截圖
# 放入 crontab 或手動執行

TARGET_DIR="$(cd "$(dirname "$0")/../backend/static/images" && pwd)"
DAYS=7

if [ -d "$TARGET_DIR" ]; then
    echo "正在清理 $TARGET_DIR 內超過 $DAYS 天的 .mp4 和 .jpg 檔案..."
    find "$TARGET_DIR" -name "*.mp4" -type f -mtime +$DAYS -exec rm -v {} \;
    find "$TARGET_DIR" -name "*.jpg" -type f -mtime +$DAYS -exec rm -v {} \;
    echo "清理完成。"
else
    echo "找不到目錄: $TARGET_DIR"
fi
