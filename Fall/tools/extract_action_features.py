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

def augment_pose_sequence(pose_seq):
    """
    對 (30, 17, 2) 的骨架序列進行資料擴增
    回傳擴增後的序列 list
    """
    augmented_seqs = []
    
    # 1. 原始序列
    augmented_seqs.append(pose_seq)
    
    # 2. 水平翻轉 (Horizontal Flip)
    # 假設 X 座標是歸一化的 [0, 1]，則 x = 1.0 - x
    flipped_seq = np.copy(pose_seq)
    flipped_seq[:, :, 0] = 1.0 - flipped_seq[:, :, 0]
    
    # 左右關鍵點對調 (YOLO Pose 17 keypoints)
    # 1: LEye, 2: REye, 3: LEar, 4: REar, 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow
    # 9: LWrist, 10: RWrist, 11: LHip, 12: RHip, 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle
    swap_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    for left_idx, right_idx in swap_pairs:
        temp = np.copy(flipped_seq[:, left_idx, :])
        flipped_seq[:, left_idx, :] = flipped_seq[:, right_idx, :]
        flipped_seq[:, right_idx, :] = temp
        
    augmented_seqs.append(flipped_seq)
    
    # 3. 微小雜訊干擾 (Jittering) 對原始
    noise_1 = np.random.normal(0, 0.005, size=pose_seq.shape)
    jittered_seq_1 = np.clip(pose_seq + noise_1, 0.0, 1.0)
    augmented_seqs.append(jittered_seq_1)
    
    # 4. 微小雜訊干擾 (Jittering) 對翻轉
    noise_2 = np.random.normal(0, 0.005, size=flipped_seq.shape)
    jittered_seq_2 = np.clip(flipped_seq + noise_2, 0.0, 1.0)
    augmented_seqs.append(jittered_seq_2)
    
    return augmented_seqs

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
                # 資料擴增：1變4
                aug_seqs = augment_pose_sequence(pose_seq)
                for seq in aug_seqs:
                    flattened_feature = seq.flatten()
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

    # Mac MPS 容易在多進程死鎖，改用安全的循序處理
    print(f"🚀 啟動循序特徵提取 (任務數: {len(tasks)})...")
    init_worker(DEFAULT_MODEL_PATH, SEQ_LENGTH)
    
    all_features = []
    all_labels = []
    
    for task in tasks:
        try:
            v_file, v_features, v_labels = process_video(task)
            all_features.extend(v_features)
            all_labels.extend(v_labels)
            print(f"✅ 完成影片: {v_file} -> 提取了 {len(v_features)} 個特徵序列")
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
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
