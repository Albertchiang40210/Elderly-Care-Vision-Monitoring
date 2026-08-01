import os
import sys
import json
import urllib.parse
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from modules.action_tracker import ActionTracker

# 設定路徑 (同時支援原本的測試影片資料夾與後端生成的事件影片資料夾)
VIDEOS_DIRS = [
    PROJECT_ROOT / "label_studio_data" / "videos",
    PROJECT_ROOT.parent / "backend" / "static" / "images"
]
OUTPUT_DIR = PROJECT_ROOT / "active_learning_dataset" / "action_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_FILE = PROJECT_ROOT / "active_learning_dataset" / "annotations.json"

DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "yolo11s-pose.pt")
CONF_THRES = 0.25
SEQ_LENGTH = 30 # 30幀作為一個動作序列

print("🔄 正在載入 ActionTracker (YOLO-Pose + Tracker)...")
tracker = ActionTracker(pose_model_path=DEFAULT_MODEL_PATH, sequence_length=SEQ_LENGTH)

all_features = []
all_labels = []

# 載入 Label Studio 標註資料
if not ANNOTATIONS_FILE.exists():
    print(f"❌ 找不到標註檔案: {ANNOTATIONS_FILE}。請先執行 fetch_annotations.py")
    sys.exit(1)

with open(ANNOTATIONS_FILE, "r") as f:
    # 處理 URL 編碼 (例如 %28 -> '(' )
    human_annotations = {urllib.parse.unquote(k): v for k, v in json.load(f).items()}

# 收集所有影片路徑
video_paths = {}
for d in VIDEOS_DIRS:
    if d.exists():
        for f in os.listdir(d):
            if f.endswith(('.mp4', '.avi', '.mov')):
                video_paths[f] = str(d / f)

print(f"🎬 總共在硬碟找到 {len(video_paths)} 部影片，準備比對人工標註...")

processed_count = 0
for video_file, video_path in video_paths.items():
    if video_file not in human_annotations:
        continue
        
    label = human_annotations[video_file]
    processed_count += 1
    print(f"[{processed_count}/{len(human_annotations)}] 處理影片: {video_file} (人工標籤: {label})")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_seqs = 0
    
    # 為了避免在一部影片中提取過多重複且高度相似的序列，我們設定每隔幾幀取樣一次 (Sliding Window Stride)
    stride = 5 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results, ready_sequences = tracker.process_frame(frame, conf_thres=CONF_THRES)
        
        if frame_count % stride == 0:
            for track_id, pose_seq in ready_sequences.items():
                # pose_seq shape: (30, 17, 2)
                # 將其展平為一維陣列 (30 * 17 * 2 = 1020 維)
                flattened_feature = pose_seq.flatten()
                all_features.append(flattened_feature)
                all_labels.append(label)
                extracted_seqs += 1
                
        frame_count += 1
        
    cap.release()
    print(f"  -> 提取了 {extracted_seqs} 個動作序列特徵")

# 儲存特徵為 NumPy 陣列 (這將被用來訓練模型)
if len(all_features) > 0:
    X = np.array(all_features)
    y = np.array(all_labels)
    output_path = OUTPUT_DIR / "action_dataset.npz"
    np.savez(output_path, X=X, y=y)
    print(f"\n✅ 特徵提取完成！")
    print(f"📊 總計提取了 {len(X)} 筆訓練資料")
    print(f"💾 特徵已儲存至: {output_path}")
    
    # 印出每個類別的數量
    unique, counts = np.unique(y, return_counts=True)
    print("類別分佈:", dict(zip(unique, counts)))
else:
    print("\n⚠️ 未能從影片中提取出任何完整的特徵序列 (可能影片太短?)")
