import time
import cv2
import os
from datetime import datetime

class RoutineSanityChecker:
    def __init__(self, camera_id, interval_seconds=21600.0):  # 🚀 業界生產規格：每 6 小時自動環境巡檢一次
        self.camera_id = camera_id
        self.interval_seconds = interval_seconds
        self.last_check_time = time.time()

    def process(self, frame, ever_detected_fall, is_leaving_bed, is_wandering, producer):
        """平常沒事、無警報時，定時截圖外發 VLM 做環境巡檢"""
        current_time = time.time()
        
        # 只有在完全沒有任何警報觸發的「閒置時間」，才啟用 VLM 巡檢大腦
        if (current_time - self.last_check_time > self.interval_seconds) and \
           not ever_detected_fall and not is_leaving_bed and not is_wandering:
            
            self.last_check_time = current_time
            
            # 🧠 核心修正 1：算出專案最外層的根目錄，精準定位主動學習隔離資料夾
            # 因為這支檔案在 modules/ 目錄下，需要跳兩層 dirname 回到專案根目錄 (Fall/)
            current_file_path = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file_path))
            vlm_save_dir = os.path.join(project_root, "active_learning_dataset", "images")
            routine_save_dir = os.path.join(project_root, "active_learning_dataset", "routine")
            os.makedirs(vlm_save_dir, exist_ok=True)
            os.makedirs(routine_save_dir, exist_ok=True)

            # 🧠 核心修正 2：組裝帶有精確時間戳的不重複檔名與絕對儲存路徑
            current_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            snapshot_name = f"snapshot_{self.camera_id}_routine_{current_time_str}.jpg"
            full_snapshot_path = os.path.join(vlm_save_dir, snapshot_name)
            routine_snapshot_path = os.path.join(routine_save_dir, snapshot_name)
            
            # 🧠 核心修正 3：雙向保存至隔離資料夾與 routine 專屬資料夾
            cv2.imwrite(full_snapshot_path, frame)
            cv2.imwrite(routine_snapshot_path, frame)

            
            try:
                # 🧠 解析出數字 ID（例如 Room_301_Bed -> 301）
                try:
                    numeric_id = int(''.join(filter(str.isdigit, self.camera_id)))
                except ValueError:
                    numeric_id = 1
                
                # 🎯 核心修正 4：Payload 同步改為傳遞絕對路徑字串
                routine_payload = {
                    "alert_id": f"RTN_{self.camera_id}_{int(current_time)}",
                    "device_id": numeric_id,                            # ✅ 對齊後端要求之 integer ID
                    "event_type": "Routine_Environment_Sanity_Check",   # ✅ 統一欄位名稱
                    "detected_at": datetime.now().isoformat(),          # ✅ 標準 ISO 時間字串
                    "camera_id": self.camera_id,                        # ✅ 統一相機 ID
                    "action_score": 1.0,                                  # ✅ 巡檢給予滿分置信度
                    
                    # 🚨 重點：將兩個路徑欄位通通綁定為隔離資料夾下的「絕對路徑」
                    # 配合智慧化改造後的 vlm_worker.py，大腦就能直接定點抓圖
                    "snapshot_path": full_snapshot_path, 
                    "image_filename": full_snapshot_path,               
                    
                    "severity": "low",                                  # ✅ 巡檢嚴重度為低
                    "status": "PENDING_VLM_ROUTE"
                }
                
                if producer is not None:
                    # 發送到大模型審查通道，榨乾 VLM 閒置算力
                    producer.send('nursing-home-alerts', value=routine_payload)
                    producer.flush()
                    print(f"🔍 [模組 G] 已發送 [{self.camera_id}] 定時巡檢截圖（已存入隔離資料夾：{snapshot_name}）至 VLM 佇列。")
                    return "Routine Checking..."
            except Exception as e:
                print(f"⚠️ [模組 G] 巡檢截圖發送失敗: {e}")
                
        return None