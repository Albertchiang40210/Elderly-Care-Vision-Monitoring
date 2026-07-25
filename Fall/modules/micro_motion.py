import numpy as np
import time
from datetime import datetime

try:
    from numba import jit
except ImportError:
    # 建立備援裝飾器，防範無 numba 的環境
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

@jit(nopython=True, fastmath=True, nogil=True)
def calculate_motion_deviation_jit(points):
    """使用 Numba JIT 加速計算時序中心點的標準差，規避 Python 迴圈慢的問題 (支援多執行緒 nogil)"""
    n = len(points)
    if n < 2:
        return 0.0
    
    # 1. 計算 X 與 Y 的均值
    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sum_x += points[i, 0]
        sum_y += points[i, 1]
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    # 2. 計算 X 與 Y 的方差
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        var_x += (points[i, 0] - mean_x) ** 2
        var_y += (points[i, 1] - mean_y) ** 2
        
    # 3. 計算 X 與 Y 的標準差
    std_x = np.sqrt(var_x / n)
    std_y = np.sqrt(var_y / n)
    
    return std_x + std_y

@jit(nopython=True, fastmath=True, nogil=True)
def get_keypoint_center_jit(kp):
    """JIT 超高速提取雙肩、雙髖與頭部核心中心點 (零記憶體配置 Zero-Allocation)"""
    indices = (0, 5, 6, 11, 12)
    sum_x = 0.0
    sum_y = 0.0
    count = 0
    for idx in indices:
        x = kp[idx, 0]
        y = kp[idx, 1]
        if x != 0.0 or y != 0.0:
            sum_x += x
            sum_y += y
            count += 1
    if count == 0:
        return np.array([0.0, 0.0], dtype=np.float64), False
    return np.array([sum_x / count, sum_y / count], dtype=np.float64), True

# 🚀 Numba 預熱 (Warmup) - 預先進行 LLVM C 編譯，防止運行時首幀微短滯後
try:
    _dummy_pts = np.zeros((30, 2), dtype=np.float64)
    _dummy_kp = np.zeros((17, 3), dtype=np.float64)
    _ = calculate_motion_deviation_jit(_dummy_pts)
    _ = get_keypoint_center_jit(_dummy_kp)
except Exception:
    pass

class MicroMotionDetector:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        # 儲存過去數影格的骨架中心點，用來算標準差
        self.motion_history = []
        self.agitation_triggered = False

    def process(self, kp, is_physically_lying, producer):
        """偵測半夜躺在床上長輩的微觀動作 (躁動偵測)"""
        if "Bed" not in self.camera_id or not is_physically_lying:
            return False

        # 🚀 使用 Numba JIT 高速提取核心骨架重心 (取代舊版 NumPy 篩選開銷)
        center_pt, has_valid = get_keypoint_center_jit(kp)
        
        if has_valid:
            self.motion_history.append(center_pt)
            if len(self.motion_history) > 45:  # 維持一個短時序視窗
                self.motion_history.pop(0)

            if len(self.motion_history) >= 30:
                # 🚀 使用 JIT 幾何標準差加速器，排除 Python numpy.std 的多餘開銷
                history_np = np.array(self.motion_history)
                total_deviation = calculate_motion_deviation_jit(history_np)

                # 💡 閾值設定：大於 0.045 代表在床上高頻率劇烈晃動、掙扎
                if total_deviation > 0.045:
                    if not self.agitation_triggered:
                        self.agitation_triggered = True
                        
                        from modules.schema import build_alert_payload

                        agitation_payload = build_alert_payload(
                            prefix="AGT",
                            camera_id=self.camera_id,
                            event_type="agitation",
                            yolo_score=total_deviation * 10,
                            vlm_summary=f"【長照預警系統：夜間身體躁動】感測到 [{self.camera_id}] 床上長輩體位出現異常高頻掙扎或躁動，疑似身體不適，請前往關懷。",
                            severity="medium"
                        )
                        
                        if producer is not None:
                            producer.send('processed-reports', value=agitation_payload)
                            producer.flush()
                            print(f"🚨 [模組 F] [{self.camera_id}] 偵測到夜間異常躁動掙扎！（規範化 Payload 對齊成功）")
                    return True
                else:
                    self.agitation_triggered = False
        return False