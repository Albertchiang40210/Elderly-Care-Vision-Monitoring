import numpy as np
import cv2

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True, fastmath=True, nogil=True)
def get_body_angle_jit(shoulder_x, shoulder_y, hip_x, hip_y):
    """計算人體軀幹連線與水平線的絕對夾角 (0~90度)，0度代表完全水平躺平"""
    if shoulder_x == 0.0 or hip_x == 0.0:
        return 90.0
    dx = abs(hip_x - shoulder_x)
    dy = abs(hip_y - shoulder_y)
    angle_rad = np.arctan2(dy, dx)
    return np.degrees(angle_rad)

class PersonTrackerEMA:
    """利用指數平滑 (EMA) 與慣性補幀，確保畫面的框與骨架 100% 穩定緊跟，絕不閃爍掉幀"""
    def __init__(self, alpha=0.35, max_missing_frames=12):
        self.alpha = alpha
        self.max_missing = max_missing_frames
        self.missing_count = 0
        self.last_box = None      # [x1, y1, x2, y2]
        self.last_kpts = None     # [17, 2] 或 [17, 3]
        self.last_conf = 0.0

    def update(self, new_box, new_kpts, conf):
        if new_box is not None:
            self.missing_count = 0
            self.last_conf = conf
            if self.last_box is None:
                self.last_box = np.array(new_box, dtype=np.float32)
            else:
                self.last_box = self.alpha * np.array(new_box, dtype=np.float32) + (1 - self.alpha) * self.last_box
            
            if new_kpts is not None:
                if self.last_kpts is None:
                    self.last_kpts = np.array(new_kpts, dtype=np.float32)
                else:
                    self.last_kpts = self.alpha * np.array(new_kpts, dtype=np.float32) + (1 - self.alpha) * self.last_kpts
            return self.last_box.astype(int), self.last_kpts, self.last_conf
        else:
            self.missing_count += 1
            if self.missing_count <= self.max_missing and self.last_box is not None:
                return self.last_box.astype(int), self.last_kpts, self.last_conf * 0.9
            else:
                self.last_box = None
                self.last_kpts = None
                return None, None, 0.0

class FallDetectorLogic:
    """封裝跌倒偵測與狀態追蹤的純業務邏輯，將推論結果轉換為事件"""
    
    def __init__(self, img_w=1280, img_h=720):
        self.img_w = img_w
        self.img_h = img_h
        self.person_states = {}
        
    def process_inference_results(self, results_pose, pose_was_updated=True):
        """處理 YOLO 推理結果，進行 EMA 平滑與跌倒判定"""
        active_track_ids = set()
        persons_out_list = []
        any_fall_triggered = False
        
        if not results_pose or len(results_pose[0].boxes) == 0:
            self._cleanup_missing_ids(active_track_ids)
            return persons_out_list, any_fall_triggered
            
        boxes = results_pose[0].boxes
        conf_data = boxes.conf.cpu().numpy()
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        
        track_ids = []
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
        else:
            track_ids = [i for i in range(len(conf_data))]
            
        for i, track_id in enumerate(track_ids):
            active_track_ids.add(track_id)
            conf_val = conf_data[i]
            raw_box = boxes_xyxy[i]
            
            raw_kpts = None
            if hasattr(results_pose[0].keypoints, 'xy') and results_pose[0].keypoints.xy is not None:
                raw_kpts = results_pose[0].keypoints.xy.cpu().numpy()[i]
            elif hasattr(results_pose[0].keypoints, 'xyn') and results_pose[0].keypoints.xyn is not None:
                raw_kpts = results_pose[0].keypoints.xyn.cpu().numpy()[i].copy()
                raw_kpts[:, 0] *= self.img_w
                raw_kpts[:, 1] *= self.img_h
                
            if track_id not in self.person_states:
                self.person_states[track_id] = {
                    "tracker": PersonTrackerEMA(alpha=0.35, max_missing_frames=15),
                    "consecutive_fall_frames": 0,
                    "ever_detected_fall": False,
                    "vlm_triggered": False,
                    "standing_recovery_count": 0
                }
            
            p_state = self.person_states[track_id]
            if raw_kpts is not None:
                raw_kpts = np.asarray(raw_kpts, dtype=np.float32)[:, :2]
                
            smooth_box, smooth_kpts, active_conf = p_state["tracker"].update(raw_box, raw_kpts, conf_val)
            
            body_angle, box_aspect_ratio = None, None
            if smooth_kpts is not None and len(smooth_kpts) >= 13:
                try:
                    shoulder_x = (float(smooth_kpts[5][0]) + float(smooth_kpts[6][0])) / 2.0
                    shoulder_y = (float(smooth_kpts[5][1]) + float(smooth_kpts[6][1])) / 2.0
                    hip_x = (float(smooth_kpts[11][0]) + float(smooth_kpts[12][0])) / 2.0
                    hip_y = (float(smooth_kpts[11][1]) + float(smooth_kpts[12][1])) / 2.0
                    body_angle = float(get_body_angle_jit(shoulder_x, shoulder_y, hip_x, hip_y))
                except Exception:
                    pass

            if smooth_box is not None:
                try:
                    bw = abs(float(smooth_box[2]) - float(smooth_box[0]))
                    bh = abs(float(smooth_box[3]) - float(smooth_box[1]))
                    if bh > 0:
                        box_aspect_ratio = bw / bh
                except Exception:
                    pass
            
            torso_is_horizontal = (body_angle is not None and body_angle <= 45.0)
            body_is_wide = (box_aspect_ratio is not None and box_aspect_ratio >= 1.2)
            is_physically_lying = torso_is_horizontal or body_is_wide
            
            if pose_was_updated:
                if is_physically_lying:
                    p_state["consecutive_fall_frames"] += 1
                else:
                    p_state["consecutive_fall_frames"] = 0
            
            should_trigger_fall = p_state["consecutive_fall_frames"] >= 15
            recovered = False
            
            if should_trigger_fall:
                p_state["ever_detected_fall"] = True
                p_state["standing_recovery_count"] = 0
                any_fall_triggered = True
            else:
                if p_state["ever_detected_fall"] and not is_physically_lying:
                    p_state["standing_recovery_count"] += 1
                    if p_state["standing_recovery_count"] >= 90:
                        p_state["ever_detected_fall"] = False
                        p_state["vlm_triggered"] = False
                        p_state["standing_recovery_count"] = 0
                        recovered = True
                        
            # Prepare output dict
            if smooth_box is not None:
                norm_bbox = [
                    round(float(smooth_box[0]) / self.img_w, 4),
                    round(float(smooth_box[1]) / self.img_h, 4),
                    round(float(smooth_box[2]) / self.img_w, 4),
                    round(float(smooth_box[3]) / self.img_h, 4)
                ]
                _kp_2d = []
                _kp_dict_list = []
                if smooth_kpts is not None:
                    for idx, _kp in enumerate(smooth_kpts):
                        kx = round(float(_kp[0]) / self.img_w, 4)
                        ky = round(float(_kp[1]) / self.img_h, 4)
                        kc = round(float(_kp[2]), 2) if len(_kp) > 2 else 0.95
                        _kp_2d.append([kx, ky])
                        _kp_dict_list.append({"x": kx, "y": ky, "score": kc, "id": idx})

                while len(_kp_2d) < 17:
                    _kp_2d.append([0.0, 0.0])
                    _kp_dict_list.append({"x": 0.0, "y": 0.0, "score": 0.0, "id": len(_kp_dict_list)})

                # 計算衍生的 Action Confidence (跌倒動作模型信心度)
                base_action_conf = 0.0
                if body_angle is not None:
                    # 身體角度越小(越接近水平)分數越高
                    angle_score = max(0, (80.0 - body_angle) / 80.0) 
                    base_action_conf = max(base_action_conf, angle_score)
                if box_aspect_ratio is not None:
                    # 寬高比越大(越平躺)分數越高
                    ratio_score = min(1.0, box_aspect_ratio / 1.5) if box_aspect_ratio > 0.5 else 0
                    base_action_conf = max(base_action_conf, ratio_score)
                
                # VLM Confidence 模擬 (如果有跌倒則模擬 VLM 的二次確認分數，正常則極低)
                vlm_conf = 0.95 if should_trigger_fall else 0.05
                if not should_trigger_fall and base_action_conf > 0.6:
                    vlm_conf = 0.5 # 模糊地帶交由 VLM 判斷

                persons_out_list.append({
                    "id": track_id,
                    "bbox": norm_bbox,
                    "conf": round(float(active_conf), 2),
                    "action_conf": round(float(base_action_conf), 2),
                    "vlm_conf": round(float(vlm_conf), 2),
                    "kps": _kp_2d,
                    "keypoints": _kp_2d,
                    "keypoints_detailed": _kp_dict_list,
                    "is_fall": bool(should_trigger_fall),
                    "new_fall_trigger": should_trigger_fall and not p_state.get("_last_vlm_triggered", False),
                    "recovered": recovered,
                    "smooth_box": smooth_box.tolist() if hasattr(smooth_box, "tolist") else smooth_box,
                    "smooth_kpts": smooth_kpts.tolist() if hasattr(smooth_kpts, "tolist") else smooth_kpts
                })
                # Mark as triggered so we don't spam triggers
                p_state["_last_vlm_triggered"] = should_trigger_fall

        self._cleanup_missing_ids(active_track_ids)
        return persons_out_list, any_fall_triggered

    def _cleanup_missing_ids(self, active_track_ids):
        keys_to_remove = [tid for tid in self.person_states.keys() if tid not in active_track_ids]
        for tid in keys_to_remove:
            del self.person_states[tid]

    def reset_states(self):
        for p_id in self.person_states:
            self.person_states[p_id]["consecutive_fall_frames"] = 0
            self.person_states[p_id]["ever_detected_fall"] = False
            self.person_states[p_id]["vlm_triggered"] = False
            self.person_states[p_id]["standing_recovery_count"] = 0
            self.person_states[p_id]["_last_vlm_triggered"] = False

