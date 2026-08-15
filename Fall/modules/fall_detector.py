import numpy as np
import cv2

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

import os
import sys
import json
import torch
import collections

# 引入 ActionTransformer 與 Settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.action_transformer import ActionTransformer
from mlops_config.settings import settings

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
        
        # === 初始化 Action Transformer ===
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
                
                # label_map 格式為 {"0": "fall", "1": "sitdown", ...}
                # 正確的 mapping 應該是 integer ID -> string label
                self.id_to_label = {int(k): v for k, v in label_map.items()}
                
                ckpt = torch.load(pt_path, map_location='cpu')
                
                # 從權重檔中動態推斷 num_classes，避免與 label_map.json 不一致導致載入失敗
                # 'classifier.3.weight' 的 shape 為 [num_classes, d_model/2]
                inferred_num_classes = len(label_map)
                if 'classifier.3.weight' in ckpt:
                    inferred_num_classes = ckpt['classifier.3.weight'].shape[0]
                
                self.action_model = ActionTransformer(
                    num_classes=inferred_num_classes,
                    input_dim=settings.INPUT_DIM,
                    d_model=settings.D_MODEL,
                    nhead=settings.NHEAD,
                    num_layers=settings.NUM_LAYERS
                )
                self.action_model.load_state_dict(ckpt)
                
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.device = torch.device('cpu')
                    
                self.action_model = self.action_model.to(self.device)
                self.action_model.eval()
                print(f"✅ [FallDetectorLogic] 成功掛載 Action Transformer (裝置: {self.device})！")
            else:
                print("⚠️ [FallDetectorLogic] 找不到 Action Transformer 模型，不回傳進階動作標籤。")
        except Exception as e:
            print(f"⚠️ [FallDetectorLogic] Action Transformer 載入失敗: {e}")
        
    def process_inference_results(self, results_pose, pose_was_updated=True):
        """處理 YOLO 推理結果，進行 EMA 平滑與跌倒判定 (加入 Batch 推論優化)"""
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
            
        # === 階段 1：計算物理特徵與準備 Batch 推論資料 ===
        batch_inputs = []
        batch_track_ids = []
        intermediate_states = {}
        
        for i, track_id in enumerate(track_ids):
            active_track_ids.add(track_id)
            conf_val = conf_data[i]
            raw_box = boxes_xyxy[i]
            
            raw_kpts = None
            if hasattr(results_pose[0].keypoints, 'data') and results_pose[0].keypoints.data is not None:
                raw_kpts = results_pose[0].keypoints.data.cpu().numpy()[i]
            elif hasattr(results_pose[0].keypoints, 'xy') and results_pose[0].keypoints.xy is not None:
                raw_kpts = results_pose[0].keypoints.xy.cpu().numpy()[i]
                
            if track_id not in self.person_states:
                self.person_states[track_id] = {
                    "tracker": PersonTrackerEMA(alpha=0.35, max_missing_frames=90),
                    "consecutive_fall_frames": 0,
                    "ever_detected_fall": False,
                    "vlm_triggered": False,
                    "standing_recovery_count": 0,
                    "kpts_sequence": collections.deque(maxlen=30),
                    "action_vote_buffer": collections.deque(maxlen=4),
                    "last_stable_action": "Tracking",
                    "prev_center": None
                }
            
            p_state = self.person_states[track_id]
            if raw_kpts is not None:
                raw_kpts = np.asarray(raw_kpts, dtype=np.float32)
                
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
            
            hip_leg_y_dist = None
            if smooth_kpts is not None and smooth_box is not None:
                try:
                    l_hip, r_hip = smooth_kpts[11], smooth_kpts[12]
                    hip_ys = []
                    if len(l_hip) > 2 and l_hip[2] > 0.3: hip_ys.append(l_hip[1])
                    if len(r_hip) > 2 and r_hip[2] > 0.3: hip_ys.append(r_hip[1])
                    
                    l_ankle, r_ankle = smooth_kpts[15], smooth_kpts[16]
                    leg_ys = []
                    if len(l_ankle) > 2 and l_ankle[2] > 0.3: leg_ys.append(l_ankle[1])
                    if len(r_ankle) > 2 and r_ankle[2] > 0.3: leg_ys.append(r_ankle[1])
                    
                    if not leg_ys:
                        l_knee, r_knee = smooth_kpts[13], smooth_kpts[14]
                        if len(l_knee) > 2 and l_knee[2] > 0.3: leg_ys.append(l_knee[1])
                        if len(r_knee) > 2 and r_knee[2] > 0.3: leg_ys.append(r_knee[1])
                    
                    if hip_ys and leg_ys:
                        hip_y = sum(hip_ys) / len(hip_ys)
                        leg_y = sum(leg_ys) / len(leg_ys)
                        bh = abs(float(smooth_box[3]) - float(smooth_box[1]))
                        if bh > 0:
                            hip_leg_y_dist = (leg_y - hip_y) / bh
                except Exception:
                    pass

            # Transformer AI 輔助：收集骨架供未來使用
            if smooth_kpts is not None:
                norm_kpts = np.copy(smooth_kpts[:, :2])
                norm_kpts[:, 0] /= self.img_w
                norm_kpts[:, 1] /= self.img_h
                p_state["kpts_sequence"].append(norm_kpts)
                
            # 決定是否加入 Batch 推論
            if self.action_model is not None and active_conf > 0.5 and len(p_state["kpts_sequence"]) >= 10:
                seq = list(p_state["kpts_sequence"])
                if len(seq) > 30:
                    seq = seq[-30:]
                elif len(seq) < 30:
                    seq = [seq[0]] * (30 - len(seq)) + seq
                
                seq_np = np.array(seq, dtype=np.float32)
                batch_inputs.append(seq_np)
                batch_track_ids.append(track_id)
                
            intermediate_states[track_id] = {
                "smooth_box": smooth_box,
                "smooth_kpts": smooth_kpts,
                "active_conf": active_conf,
                "body_angle": body_angle,
                "box_aspect_ratio": box_aspect_ratio,
                "hip_leg_y_dist": hip_leg_y_dist,
                "pred_label_transformer": None,
                "pred_conf_val": 0.0
            }

        # === 階段 2：Batch AI 推論 ===
        if batch_inputs and hasattr(self, "device"):
            batch_tensor = torch.from_numpy(np.stack(batch_inputs)).to(self.device)
            try:
                with torch.no_grad():
                    logits = self.action_model(batch_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    pred_idxs = torch.argmax(probs, dim=-1).cpu().numpy()
                    pred_confs = probs[torch.arange(len(probs)), pred_idxs].cpu().numpy()
                    
                    for idx, tid in enumerate(batch_track_ids):
                        pred_idx = pred_idxs[idx]
                        intermediate_states[tid]["pred_label_transformer"] = self.id_to_label.get(pred_idx)
                        intermediate_states[tid]["pred_conf_val"] = float(pred_confs[idx])
            except Exception as e:
                pass
                
        # === 階段 3：整合物理規則與判定 ===
        for track_id in track_ids:
            istate = intermediate_states[track_id]
            p_state = self.person_states[track_id]
            
            smooth_box = istate["smooth_box"]
            smooth_kpts = istate["smooth_kpts"]
            active_conf = istate["active_conf"]
            body_angle = istate["body_angle"]
            box_aspect_ratio = istate["box_aspect_ratio"]
            hip_leg_y_dist = istate["hip_leg_y_dist"]
            pred_label_transformer = istate["pred_label_transformer"]
            pred_conf_val = istate["pred_conf_val"]
            
            torso_is_horizontal = (body_angle is not None and body_angle <= 45.0)
            body_is_wide = (box_aspect_ratio is not None and box_aspect_ratio >= 1.2)
            
            if pred_label_transformer == "fall" and hip_leg_y_dist is not None and hip_leg_y_dist > 0.25:
                pred_label_transformer = "normal"
                
            if body_angle is not None and body_angle > 55.0:
                if pred_label_transformer == "fall":
                    pred_label_transformer = "normal"
                
            if hip_leg_y_dist is not None and hip_leg_y_dist > 0.20:
                body_is_wide = False
                torso_is_horizontal = False
                
            is_crumpled = False
            if hip_leg_y_dist is not None:
                is_crumpled = (-0.1 <= hip_leg_y_dist <= 0.08)
                
            is_confident_sit = (pred_label_transformer == "sitdown" and pred_conf_val > 0.5) or (body_angle is not None and body_angle > 55.0)
            
            is_physically_lying = (torso_is_horizontal or body_is_wide or is_crumpled) and not is_confident_sit
            is_ai_fall = (pred_label_transformer == "fall" and pred_conf_val > 0.85) and not is_confident_sit
            
            if pose_was_updated:
                if is_physically_lying or is_ai_fall:
                    p_state["consecutive_fall_frames"] += 1
                else:
                    p_state["consecutive_fall_frames"] = max(0, p_state["consecutive_fall_frames"] - 2)
            
            fall_threshold = 6 if is_ai_fall else 12
            should_trigger_fall = p_state["consecutive_fall_frames"] >= fall_threshold
            recovered = False
            
            if should_trigger_fall:
                p_state["ever_detected_fall"] = True
                p_state["standing_recovery_count"] = 0
                any_fall_triggered = True
            else:
                if p_state["ever_detected_fall"] and not is_physically_lying:
                    p_state["standing_recovery_count"] += 1
                    if p_state["standing_recovery_count"] >= 150:
                        p_state["ever_detected_fall"] = False
                        p_state["vlm_triggered"] = False
                        p_state["standing_recovery_count"] = 0
                        recovered = True
                        
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

                base_action_conf = pred_conf_val if pred_label_transformer else 0.0
                if body_angle is not None:
                    angle_score = max(0, (80.0 - body_angle) / 80.0) 
                    base_action_conf = max(base_action_conf, angle_score)
                if box_aspect_ratio is not None:
                    ratio_score = min(1.0, box_aspect_ratio / 1.5) if box_aspect_ratio > 0.5 else 0
                    base_action_conf = max(base_action_conf, ratio_score)
                
                vlm_conf = 0.95 if should_trigger_fall else 0.05
                if not should_trigger_fall and base_action_conf > 0.6:
                    vlm_conf = 0.5

                if should_trigger_fall or (p_state["ever_detected_fall"] and p_state["standing_recovery_count"] < 150):
                    action_label = "Fall"
                else:
                    action_label = "Normal"
                    if pred_label_transformer is not None and pred_conf_val > 0.5:
                        transformer_mapping = {
                            "fall": "Fall",
                            "sitdown": "Sitting",
                            "squat": "Squatting",
                            "walk": "Walking",
                            "normal": "Normal",
                            "kneel": "Kneeling"
                        }
                        action_label = transformer_mapping.get(pred_label_transformer, "Normal")
                    
                    if body_angle is not None and box_aspect_ratio is not None:
                        if body_angle >= 68.0:
                            cx = (float(smooth_box[0]) + float(smooth_box[2])) / 2
                            cy = (float(smooth_box[1]) + float(smooth_box[3])) / 2
                            bw = abs(float(smooth_box[2]) - float(smooth_box[0]))
                            bh = abs(float(smooth_box[3]) - float(smooth_box[1]))
                            is_moving = False
                            
                            if p_state["prev_center"] is not None:
                                if len(p_state["prev_center"]) >= 4:
                                    dx = abs(cx - p_state["prev_center"][0])
                                    dy = abs(cy - p_state["prev_center"][1])
                                    dw = abs(bw - p_state["prev_center"][2])
                                    dh = abs(bh - p_state["prev_center"][3])
                                    if (dx + dy > 3.0) or (dw + dh > 2.0):
                                        is_moving = True
                                elif len(p_state["prev_center"]) == 2:
                                    dx = abs(cx - p_state["prev_center"][0])
                                    dy = abs(cy - p_state["prev_center"][1])
                                    if dx + dy > 3.0:
                                        is_moving = True
                                    
                            p_state["prev_center"] = (cx, cy, bw, bh)
                            if is_moving and action_label in ["Normal", "Sitting"]:
                                action_label = "Walking"
                                
                        if action_label == "Sitting" and box_aspect_ratio < 0.55 and body_angle > 55.0:
                            action_label = "Bending" if body_angle < 70.0 else "Normal"
                                
                        if action_label == "Normal":
                            if body_angle < 60.0 and body_angle >= 35.0:
                                action_label = "Bending"
                            elif box_aspect_ratio > 0.6 and body_angle < 85.0:
                                action_label = "Sitting"
                            elif body_angle < 35.0:
                                action_label = "Fall" if box_aspect_ratio > 1.0 else "Bending"
                
                p_state["action_vote_buffer"].append(action_label)
                if len(p_state["action_vote_buffer"]) >= 3:
                    if should_trigger_fall or action_label == "Fall":
                        final_action_label = "Fall"
                    else:
                        from collections import Counter
                        votes = Counter(p_state["action_vote_buffer"])
                        final_action_label = votes.most_common(1)[0][0]
                else:
                    final_action_label = action_label
                    
                p_state["last_stable_action"] = final_action_label

                persons_out_list.append({
                    "id": track_id,
                    "bbox": norm_bbox,
                    "conf": round(float(active_conf), 2),
                    "action_conf": round(float(base_action_conf), 2),
                    "vlm_conf": round(float(vlm_conf), 2),
                    "action": final_action_label,
                    "kps": _kp_2d,
                    "keypoints": _kp_2d,
                    "keypoints_detailed": _kp_dict_list,
                    "is_fall": bool(should_trigger_fall),
                    "new_fall_trigger": should_trigger_fall and not p_state.get("_last_vlm_triggered", False),
                    "recovered": recovered,
                    "smooth_box": smooth_box.tolist() if hasattr(smooth_box, "tolist") else smooth_box,
                    "smooth_kpts": smooth_kpts.tolist() if hasattr(smooth_kpts, "tolist") else smooth_kpts
                })
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

