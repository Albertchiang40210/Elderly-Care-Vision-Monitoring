import os
import sys
import json
import urllib.parse
from pathlib import Path
import cv2
import numpy as np
import concurrent.futures

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from mlops_config.settings import settings
from modules.action_tracker import ActionTracker

# 設定路徑
VIDEOS_DIRS = [
    PROJECT_ROOT / "label_studio_data" / "videos",
    PROJECT_ROOT.parent / "backend" / "static" / "images"
]
OUTPUT_DIR = PROJECT_ROOT / "active_learning_dataset" / "action_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_FILE = PROJECT_ROOT / "active_learning_dataset" / "annotations.json"

DEFAULT_MODEL_PATH = str(settings.POSE_MODEL_PATH)
CONF_THRES = settings.CONF_THRES
SEQ_LENGTH = settings.SEQ_LENGTH

# 全域變數供子進程快取模型
_local_tracker = None

def init_worker(model_path, seq_len):
    """子進程初始化時只載入一次 YOLO 模型，避免重複載入耗時"""
    global _local_tracker
    # 關閉 print 以免畫面太亂
    import sys, os
    sys.stdout = open(os.devnull, 'w')
    _local_tracker = ActionTracker(pose_model_path=model_path, sequence_length=seq_len)
    sys.stdout = sys.__stdout__

def process_video(video_info):
    """處理單部影片的任務函數"""
    video_file, video_path, label, conf_thres = video_info
    global _local_tracker
    
    features = []
    labels = []
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    stride = settings.STRIDE 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results, ready_sequences = _local_tracker.process_frame(frame, conf_thres=conf_thres)
        
        if frame_count % stride == 0:
            for track_id, pose_seq in ready_sequences.items():
                flattened_feature = pose_seq.flatten()
                features.append(flattened_feature)
                labels.append(label)
                
        frame_count += 1
        
    cap.release()
    return video_file, features, labels

def main():
    if not ANNOTATIONS_FILE.exists():
        print(f"❌ 找不到標註檔案: {ANNOTATIONS_FILE}。請先執行 fetch_annotations.py")
        sys.exit(1)

    with open(ANNOTATIONS_FILE, "r") as f:
        human_annotations = {urllib.parse.unquote(k): v for k, v in json.load(f).items()}

    video_paths = {}
    for d in VIDEOS_DIRS:
        if d.exists():
            for f in os.listdir(d):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    video_paths[f] = str(d / f)

    print(f"🎬 總共在硬碟找到 {len(video_paths)} 部影片，準備比對人工標註...")

    tasks = []
    for video_file, video_path in video_paths.items():
        if video_file in human_annotations:
            tasks.append((video_file, video_path, human_annotations[video_file], CONF_THRES))

    print(f"🚀 啟動多進程特徵提取 (任務數: {len(tasks)})...")
    
    all_features = []
    all_labels = []
    processed_count = 0
    
    # 根據 CPU 核心數自動決定 workers 數量，但不超過 4 以免記憶體爆炸
    max_workers = min(4, (os.cpu_count() or 2)) 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(DEFAULT_MODEL_PATH, SEQ_LENGTH)) as executor:
        futures = [executor.submit(process_video, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                v_file, v_features, v_labels = future.result()
                processed_count += 1
                print(f"[{processed_count}/{len(tasks)}] 完成影片: {v_file} -> 提取了 {len(v_features)} 個特徵序列")
                all_features.extend(v_features)
                all_labels.extend(v_labels)
            except Exception as e:
                print(f"❌ 處理影片發生錯誤: {e}")

    if len(all_features) > 0:
        X = np.array(all_features)
        y = np.array(all_labels)
        output_path = OUTPUT_DIR / "action_dataset.npz"
        np.savez(output_path, X=X, y=y)
        print(f"\n✅ 平行特徵提取完成！耗時大幅縮短！")
        print(f"📊 總計提取了 {len(X)} 筆訓練資料")
        print(f"💾 特徵已儲存至: {output_path}")
        
        unique, counts = np.unique(y, return_counts=True)
        print("類別分佈:", dict(zip(unique, counts)))
    else:
        print("\n⚠️ 未能從影片中提取出任何完整的特徵序列")

if __name__ == '__main__':
    main()
