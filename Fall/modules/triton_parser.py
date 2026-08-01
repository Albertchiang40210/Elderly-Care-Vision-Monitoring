import torch
import numpy as np

class MockBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor([xyxy])
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([cls])
        self.id = None # for tracking

class MockResults:
    def __init__(self, boxes):
        self.boxes = boxes

class MockPoseKeypoints:
    def __init__(self, xyn):
        self.xyn = torch.tensor(xyn)
    def __len__(self):
        return len(self.xyn)

class MockPoseBoxes:
    def __init__(self, conf, xywh, xyxy):
        self.conf = torch.tensor(conf)
        self.xywh = torch.tensor(xywh)
        self.xyxy = torch.tensor(xyxy)
        self.id = None
    def __len__(self):
        return len(self.conf)

class MockPoseResults:
    def __init__(self, keypoints, boxes, original_frame):
        self.keypoints = keypoints
        self.boxes = boxes
        self.original_frame = original_frame

def parse_triton_yolo_pose(raw_output, img_w, img_h, conf_threshold=0.08, iou_threshold=0.45):
    """將 Triton Server 回傳的 YOLO-Pose 原始 Tensor 轉換為相容的資料格式"""
    data = raw_output[0].T
    if data.ndim != 2 or data.shape[1] < 56:
        return None
    scores = data[:, 4]
    valid_indices = np.where(scores >= conf_threshold)[0]
    if len(valid_indices) == 0:
        return None
        
    valid_data = data[valid_indices]
    valid_scores = scores[valid_indices]
    
    cx, cy, w, h = valid_data[:, 0], valid_data[:, 1], valid_data[:, 2], valid_data[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    x1_b, y1_b, x2_b, y2_b = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2_b - x1_b) * (y2_b - y1_b)
    order = valid_scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1_b[i], x1_b[order[1:]])
        yy1 = np.maximum(y1_b[i], y1_b[order[1:]])
        xx2 = np.minimum(x2_b[i], x2_b[order[1:]])
        yy2 = np.minimum(y2_b[i], y2_b[order[1:]])
        
        w_int = np.maximum(0.0, xx2 - xx1)
        h_int = np.maximum(0.0, yy2 - yy1)
        inter = w_int * h_int
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    if len(keep) == 0:
        return None
        
    final_boxes_xyxy = boxes[keep]
    final_scores = valid_scores[keep]
    final_data = valid_data[keep]
    
    scale_x = img_w / 640.0
    scale_y = img_h / 640.0
    
    final_boxes_xyxy[:, [0, 2]] *= scale_x
    final_boxes_xyxy[:, [1, 3]] *= scale_y
    
    kpts_raw = final_data[:, 5:]
    N = kpts_raw.shape[0]
    kpt_scores = kpts_raw[:, 2::3]
    pose_is_valid = np.count_nonzero(kpt_scores >= 0.20, axis=1) >= 6
    if not np.any(pose_is_valid):
        return None

    final_boxes_xyxy = final_boxes_xyxy[pose_is_valid]
    final_scores = final_scores[pose_is_valid]
    final_data = final_data[pose_is_valid]
    kpts_raw = kpts_raw[pose_is_valid]
    N = kpts_raw.shape[0]
    kpts_xyn = np.zeros((N, 17, 3), dtype=np.float32)
    for k in range(17):
        kpts_xyn[:, k, 0] = kpts_raw[:, k*3] / 640.0
        kpts_xyn[:, k, 1] = kpts_raw[:, k*3 + 1] / 640.0
        kpts_xyn[:, k, 2] = kpts_raw[:, k*3 + 2]
        
    final_cx = final_data[:, 0] * scale_x
    final_cy = final_data[:, 1] * scale_y
    final_w = final_data[:, 2] * scale_x
    final_h = final_data[:, 3] * scale_y
    final_boxes_xywh = np.stack([final_cx, final_cy, final_w, final_h], axis=1)
    
    return {
        "xyn": kpts_xyn,
        "conf": final_scores,
        "xywh": final_boxes_xywh,
        "xyxy": final_boxes_xyxy
    }
