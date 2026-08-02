import os
import sys
import collections
import numpy as np
from ultralytics import YOLO
import torch
import json

# 加入根目錄到 sys.path 確保能匯入其他模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from modules.action_transformer import ActionTransformer
    from mlops_config.settings import settings
except ImportError:
    pass

class ActionTracker:
    def __init__(self, pose_model_path: str, sequence_length: int = 30):
        """
        初始化動作追蹤器
        """
        self.model = YOLO(pose_model_path)
        self.sequence_length = sequence_length
        self.track_history = collections.defaultdict(
            lambda: collections.deque(maxlen=self.sequence_length)
        )
        
        # 嘗試載入 Phase 3 訓練完的 Action Transformer 模型
        self.action_model = None
        self.id_to_label = {}
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(project_root, "models", "action_classifier")
            pt_path = os.path.join(model_dir, "transformer_action_model.pt")
            label_map_path = os.path.join(model_dir, "label_map.json")
            
            if os.path.exists(pt_path) and os.path.exists(label_map_path):
                with open(label_map_path, "r") as f:
                    label_map = json.load(f)
                self.id_to_label = {v: k for k, v in label_map.items()}
                
                self.action_model = ActionTransformer(
                    num_classes=len(label_map),
                    input_dim=settings.INPUT_DIM,
                    d_model=settings.D_MODEL,
                    nhead=settings.NHEAD,
                    num_layers=settings.NUM_LAYERS
                )
                self.action_model.load_state_dict(torch.load(pt_path, map_location='cpu'))
                self.action_model.eval()
                print("✅ [ActionTracker] 已成功掛載 Action Transformer AI 模型大腦！")
            else:
                print("⚠️ [ActionTracker] 找不到訓練好的模型檔，將降級使用啟發式規則。")
        except Exception as e:
            print(f"⚠️ [ActionTracker] 模型載入失敗: {e}")
        
    def process_frame(self, frame, conf_thres: float = 0.25):
        results = self.model.track(frame, persist=True, conf=conf_thres, verbose=False, tracker="bytetrack.yaml")
        ready_sequences = {}
        
        if len(results) > 0 and results[0].boxes.id is not None and results[0].keypoints is not None:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            track_ids = boxes.id.int().cpu().tolist()
            kpts_xyn = keypoints.xyn.cpu().numpy()
            
            for track_id, kpt in zip(track_ids, kpts_xyn):
                self.track_history[track_id].append(kpt)
                if len(self.track_history[track_id]) == self.sequence_length:
                    ready_sequences[track_id] = np.array(self.track_history[track_id])
                    
        return results[0], ready_sequences

    def predict_action(self, pose_sequence: np.ndarray):
        """
        呼叫時間序列分類器 (Phase 3 真實推理)
        """
        if self.action_model is not None:
            # pose_sequence 形狀為 (seq_len, 17, 2)
            seq_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0) # (1, seq_len, 17, 2)
            batch_size, seq_len, num_kpts, coords = seq_tensor.shape
            seq_tensor = seq_tensor.view(batch_size, seq_len, num_kpts * coords) # 攤平為 (1, seq_len, 34)
            
            with torch.no_grad():
                logits = self.action_model(seq_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class_idx = torch.argmax(probs, dim=1).item()
                
            return self.id_to_label.get(pred_class_idx, "normal")
            
        else:
            # 降級使用 Heuristic Placeholder
            latest_pose = pose_sequence[-1]
            nose_y = latest_pose[0][1]
            left_ankle_y = latest_pose[15][1]
            right_ankle_y = latest_pose[16][1]
            
            if nose_y > 0 and left_ankle_y > 0 and right_ankle_y > 0:
                if nose_y > left_ankle_y and nose_y > right_ankle_y:
                    return "fall"
            return "normal"
