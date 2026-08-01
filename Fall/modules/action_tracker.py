import collections
import numpy as np
from ultralytics import YOLO

class ActionTracker:
    def __init__(self, pose_model_path: str, sequence_length: int = 30):
        """
        初始化動作追蹤器
        :param pose_model_path: YOLO Pose 模型權重路徑 (e.g., yolo11s-pose.pt)
        :param sequence_length: 收集多少個影格的特徵作為一個序列，用來判斷動作 (預設 30 幀，約 1 秒)
        """
        self.model = YOLO(pose_model_path)
        self.sequence_length = sequence_length
        # 使用 dictionary 儲存每個 track_id 的歷史 pose 序列
        # deque 會自動保持最大長度為 sequence_length
        self.track_history = collections.defaultdict(
            lambda: collections.deque(maxlen=self.sequence_length)
        )
        
    def process_frame(self, frame, conf_thres: float = 0.25):
        """
        處理單個影格：進行物件追蹤與骨架提取
        :param frame: cv2 讀取的影像 (numpy array)
        :param conf_thres: 信心度門檻
        :return: (預測結果, 包含 sequence 達標的 tracked_persons)
        """
        # 使用 Ultralytics 內建的 ByteTrack 追蹤
        # persist=True 表示保持跨 frame 的追蹤狀態
        results = self.model.track(frame, persist=True, conf=conf_thres, verbose=False, tracker="bytetrack.yaml")
        
        ready_sequences = {}
        
        if len(results) > 0 and results[0].boxes.id is not None and results[0].keypoints is not None:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            track_ids = boxes.id.int().cpu().tolist()
            kpts_xyn = keypoints.xyn.cpu().numpy() # shape: [N, 17, 2]
            
            for track_id, kpt in zip(track_ids, kpts_xyn):
                # 將當前影格的骨架關鍵點存入該 ID 的歷史紀錄
                self.track_history[track_id].append(kpt)
                
                # 如果收集到的歷史長度達標，可以丟給 Action Classifier
                if len(self.track_history[track_id]) == self.sequence_length:
                    ready_sequences[track_id] = np.array(self.track_history[track_id])
                    
        return results[0], ready_sequences

    def predict_action(self, pose_sequence: np.ndarray):
        """
        呼叫時間序列分類器 (Phase 3 將會訓練這個模型)
        :param pose_sequence: shape [sequence_length, 17, 2] 的 numpy array
        :return: 動作類別字串 (例如 "fall", "normal")
        """
        # TODO: 這裡在 Phase 3 訓練完模型後，將會載入 PyTorch 或 XGBoost 模型進行推理
        # 目前暫時回傳 "normal" 作為 Placeholder
        
        # 簡單的 Heuristic Placeholder: 
        # 檢查鼻子(0)和腳踝(15,16)的 Y 座標相對位置，若鼻子低於腳踝，可能是跌倒
        # 由於 y 向下為正，y 值越大表示在畫面越下方
        latest_pose = pose_sequence[-1]
        nose_y = latest_pose[0][1]
        left_ankle_y = latest_pose[15][1]
        right_ankle_y = latest_pose[16][1]
        
        if nose_y > 0 and left_ankle_y > 0 and right_ankle_y > 0:
            if nose_y > left_ankle_y and nose_y > right_ankle_y:
                return "fall"
                
        return "normal"
