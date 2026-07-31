import os
import sys
import json
import shutil
import requests
from pathlib import Path

# =========================================================================
# 1. 參數配置
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

load_dotenv(BASE_DIR / ".env")
if not os.environ.get("LS_URL"):
    load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8082")
USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")
PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "")

PID = 1  # Fall_Detection project ID

DATASET_DIR = PROJECT_ROOT / "active_learning_pose_dataset"
LABELS_DIR = DATASET_DIR / "labels"
IMAGES_DIR = DATASET_DIR / "images"

# 來源圖片資料夾 (False alarms)
SOURCE_IMAGES_DIR = PROJECT_ROOT / "label_studio_data" / "pose_false_alarms" / "images"

LABELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# =========================================================================
# 2. 登入 Label Studio
# =========================================================================
print(f"[*] 正在連線至 Label Studio 下載 YOLO-Pose 人工標註資料...")
session = requests.Session()
login_page_url = f"{LS_URL}/user/login/"
try:
    session.get(login_page_url, timeout=5)
    csrftoken = session.cookies.get('csrftoken', '')
    login_data = {"email": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken}
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": login_page_url})
    session.post(login_page_url, data=login_data, allow_redirects=True)
    session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
except Exception as e:
    print(f"❌ 連線失敗: {e}")
    sys.exit(1)

# =========================================================================
# 3. 獲取 Tasks 並解析
# =========================================================================
tasks_res = session.get(f"{LS_URL}/api/tasks/?project={PID}&page_size=1000", timeout=10)
if tasks_res.status_code != 200:
    print(f"❌ 獲取 Tasks 失敗: HTTP {tasks_res.status_code}")
    sys.exit(1)

tasks = tasks_res.json()
if isinstance(tasks, dict):
    tasks = tasks.get("tasks", [])

exported_count = 0

for task in tasks:
    if not task.get("is_labeled"): continue
    task_id = task.get("id")
    
    ann_res = session.get(f"{LS_URL}/api/tasks/{task_id}/annotations/", timeout=5)
    if ann_res.status_code != 200: continue
    
    annotations = ann_res.json()
    if not annotations:
        continue  # 沒有人工確認的標註
    
    # 取最新的標註
    latest_annotation = annotations[-1]
    results = latest_annotation.get("result", [])
    if not results:
        continue
    
    # 圖片檔名解析
    file_upload = task.get("file_upload") or task.get("data", {}).get("image", "")
    if not file_upload: continue
    filename = os.path.basename(file_upload).split("?")[0]
    
    # 根據 person ID 分組 keypoints
    persons = {}
    
    for res in results:
        if res.get("type") == "keypointlabels":
            kp_labels = res["value"].get("keypointlabels", [])
            if not kp_labels: continue
            
            kp_name = kp_labels[0]
            if kp_name not in KPT_NAMES: continue
            
            kp_idx = KPT_NAMES.index(kp_name)
            x = res["value"]["x"] / 100.0
            y = res["value"]["y"] / 100.0
            
            # 從 id 提取 person_idx，例如 "kp_0_1"
            res_id = str(res.get("id", ""))
            parts = res_id.split("_")
            person_idx = parts[1] if len(parts) >= 2 else "0"
            
            if person_idx not in persons:
                persons[person_idx] = {i: [0.0, 0.0, 0] for i in range(17)}
            
            persons[person_idx][kp_idx] = [x, y, 2] # visibility = 2 (labeled)
            
    if not persons:
        continue
        
    # 生成 YOLO-Pose 檔案內容
    yolo_lines = []
    for p_idx, kpts in persons.items():
        # 計算 bounding box
        valid_x = [k[0] for k in kpts.values() if k[2] > 0]
        valid_y = [k[1] for k in kpts.values() if k[2] > 0]
        
        if not valid_x or not valid_y: continue
        
        min_x, max_x = min(valid_x), max(valid_x)
        min_y, max_y = min(valid_y), max(valid_y)
        
        # 加上 5% 的 padding
        pad_x = (max_x - min_x) * 0.05
        pad_y = (max_y - min_y) * 0.05
        
        box_x = max(0.0, min_x - pad_x)
        box_y = max(0.0, min_y - pad_y)
        box_w = min(1.0, max_x + pad_x) - box_x
        box_h = min(1.0, max_y + pad_y) - box_y
        
        cx = box_x + box_w / 2
        cy = box_y + box_h / 2
        
        line = [0, cx, cy, box_w, box_h]
        for i in range(17):
            line.extend(kpts[i])
            
        yolo_lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in line))
        
    if yolo_lines:
        # 寫入 txt
        stem = Path(filename).stem
        txt_path = LABELS_DIR / f"{stem}.txt"
        txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        
        # 複製圖片
        src_img = SOURCE_IMAGES_DIR / filename
        dst_img = IMAGES_DIR / filename
        if src_img.exists():
            shutil.copy(src_img, dst_img)
        
        exported_count += 1

print(f"✅ 成功下載並轉換 {exported_count} 筆人工標註至 YOLO-Pose 格式！")
