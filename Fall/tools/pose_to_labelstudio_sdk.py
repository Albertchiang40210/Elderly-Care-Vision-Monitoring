import os
import sys
from pathlib import Path
import cv2
import requests
import json
import numpy as np
import shutil
import time  
import base64  
from ultralytics import YOLO

# =========================================================================
# 1. 參數與環境變數配置區
# =========================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

def load_dotenv(path: Path) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

# 🎯 環境變數強注
env_file_path = BASE_DIR / ".env"
if env_file_path.exists():
    load_dotenv(env_file_path)
else:
    load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8080") 
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))

USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")  
PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "")     

# 骨架模型路徑
ACTIVE_MODEL_PATH = str(PROJECT_ROOT / "active_learning_pose_dataset" / "models" / "yolo_pose" / "best.pt")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "yolo11s-pose.pt")
MODEL_PATH = ACTIVE_MODEL_PATH if os.path.exists(ACTIVE_MODEL_PATH) else DEFAULT_MODEL_PATH

# 🎯 指定抓取 VLM 判定為「誤報」的跌倒資料夾 (專屬 Label Studio 隔離區)
IMAGES_DIR = PROJECT_ROOT / "label_studio_data" / "pose_false_alarms" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# COCO 17 個關節點對應表
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def fail(msg: str) -> None:
    print(f"\n[X] {msg}")
    sys.exit(1)

# =========================================================================
# 2. 模擬瀏覽器登入 Label Studio
# =========================================================================
print(f"[*] 正在建立 Session 並嘗試登入 {LS_URL} ...")
session = requests.Session()
login_page_url = f"{LS_URL}/user/login/"
try:
    init_res = session.get(login_page_url, timeout=5)
    csrftoken = session.cookies.get('csrftoken', '')
except Exception as e: fail(f"無法連線至 Label Studio 服務: {e}")

login_data = {"email": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken}
session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "Referer": login_page_url})
login_res = session.post(login_page_url, data=login_data, allow_redirects=True)
if "login" in login_res.url: fail("帳號或密碼錯誤，請檢查 USERNAME 和 PASSWORD 設定！")
print("🎉 [登入成功] 已獲取合法網頁 Session 憑證！")

# =========================================================================
# 3. 讀取模型並自動執行全專案 AI 預標註 (Predictions) - Keypoints
# =========================================================================
print(f"🔄 正在載入 YOLO-Pose 模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 只鎖定跌倒/骨架專案
TARGET_PROJECTS = [{"id": 1, "name": "Fall_Detection"}]

total_pushed = 0

for p_target in TARGET_PROJECTS:
    pid = p_target["id"]
    pname = p_target["name"]
    print(f"\n⏳ [智慧預打標] 正在讀取專案 {pid} ({pname}) 的 Tasks 並發動預測標註...")
    
    FROM_NAME_KP, TO_NAME = "kp-1", "image"
    
    # 🎯 同步 Local Storage
    try:
        storage_url = f"{LS_URL}/api/storages/localfiles/?project={pid}"
        storage_res = session.get(storage_url, timeout=5)
        if storage_res.status_code == 200:
            storages = storage_res.json()
            for st in storages:
                st_id = st.get("id")
                sync_url = f"{LS_URL}/api/storages/localfiles/{st_id}/sync"
                session.post(sync_url, timeout=5)
                time.sleep(1)
    except Exception: pass

    # 取得 Tasks
    tasks_res = session.get(f"{LS_URL}/api/tasks/?project={pid}", timeout=10)
    if tasks_res.status_code != 200: continue
    tasks = tasks_res.json()
    
    if not isinstance(tasks, list):
        tasks = tasks.get("tasks", []) if isinstance(tasks, dict) else []

    unannotated_tasks = [t for t in tasks if not t.get("is_labeled") and not t.get("predictions")]
    if not unannotated_tasks:
        print(f"✅ 專案 {pname} 目前沒有需要 AI 預標籤的新照片。")
        continue

    print(f"👁️ 發現 {len(unannotated_tasks)} 張未標註圖片，啟動 YOLO-Pose 骨架推論...")

    for task in unannotated_tasks:
        task_id = task.get("id")
        file_upload = task.get("file_upload") or task.get("data", {}).get("image")
        if not file_upload: continue
        
        filename = os.path.basename(file_upload).split("?")[0]
        local_img_path = IMAGES_DIR / filename

        if not local_img_path.exists():
            print(f"⚠️ 找不到本地圖片: {local_img_path}")
            continue

        try:
            results = model.predict(source=str(local_img_path), conf=CONF_THRES, verbose=False)
            task_predictions = []
            
            for result in results:
                if result.keypoints is None or result.keypoints.xyn is None: continue
                
                width, height = int(result.orig_shape[1]), int(result.orig_shape[0])
                kpts_xyn = result.keypoints.xyn.cpu().numpy()  # [N, 17, 2]
                confs = result.boxes.conf.cpu().numpy()
                
                # 對畫面中的每個人 (person) 進行骨架標註
                for person_idx, kpt_person in enumerate(kpts_xyn):
                    person_conf = float(confs[person_idx])
                    
                    for kp_idx, (x_norm, y_norm) in enumerate(kpt_person):
                        if x_norm == 0 and y_norm == 0: continue # 該關節點被遮擋或無效
                        
                        kp_name = KPT_NAMES[kp_idx]
                        x_pct, y_pct = float(x_norm * 100), float(y_norm * 100)
                        
                        res_item = {
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "value": {
                                "x": x_pct,
                                "y": y_pct,
                                "width": 0.5,
                                "keypointlabels": [kp_name]
                            },
                            "id": f"kp_{person_idx}_{kp_idx}",
                            "from_name": FROM_NAME_KP,
                            "to_name": TO_NAME,
                            "type": "keypointlabels",
                            "score": person_conf
                        }
                        task_predictions.append(res_item)

            if task_predictions:
                payload = {
                    "task": task_id,
                    "model_version": Path(MODEL_PATH).name,
                    "result": task_predictions
                }
                
                pred_url = f"{LS_URL}/api/predictions/"
                post_res = session.post(pred_url, json=payload, timeout=5)
                if post_res.status_code in [200, 201]:
                    total_pushed += 1
                else:
                    print(f"❌ 任務 {task_id} 預標籤寫入失敗: HTTP {post_res.status_code}")
                    
        except Exception as e:
            print(f"⚠️ 處理圖片 {filename} 時發生錯誤: {e}")

print(f"\n🎉 完美收工！本次骨架辨識共成功推送 {total_pushed} 張預標註至 Label Studio！\n")
