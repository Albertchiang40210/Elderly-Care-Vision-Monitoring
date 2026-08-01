#!/bin/bash
set -e

# 取得腳本所在的目錄 (Fall)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "🚀 啟動動作分類器 (Action Transformer) 全自動重訓流程..."

# 啟用虛擬環境
source .venv/bin/activate

echo ""
echo "=========================================="
echo "1️⃣ 從 Label Studio 抓取最新人工標註..."
echo "=========================================="
python tools/fetch_annotations.py

echo ""
echo "=========================================="
echo "2️⃣ 進行骨架特徵提取 (YOLO-Pose)..."
echo "=========================================="
python tools/extract_action_features.py

echo ""
echo "=========================================="
echo "3️⃣ 訓練 Action Transformer (擂台賽對決)..."
echo "=========================================="
python tools/train_action_classifier.py

echo ""
echo "=========================================="
echo "4️⃣ 匯出 ONNX 並熱更新至模型庫..."
echo "=========================================="
python tools/export_action_model.py

echo ""
echo "=========================================="
echo "🎉 全自動重訓與熱更新完成！前端警衛已換上新大腦！"
echo "=========================================="
