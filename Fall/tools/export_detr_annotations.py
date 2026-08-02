import os
import sys
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

PID = 2  # Hazard_Detection project ID

DATASET_DIR = PROJECT_ROOT / "active_learning_dataset"
LABELS_DIR = DATASET_DIR / "labels"
IMAGES_DIR = DATASET_DIR / "images"

# 來源圖片資料夾 (DETR false alarms)
SOURCE_IMAGES_DIR = PROJECT_ROOT / "label_studio_data" / "false_alarms" / "images"

LABELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ENV_LABEL_TO_YOLO = {
    "wheelchair": 0,
    "bed": 1,
    "person": 2,
    "obstacle": 3
}

# =========================================================================
# 2. 登入 Label Studio
# =========================================================================
print(f"[*] 正在連線至 Label Studio 下載 DETR 人工標註資料...")
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
tasks = []
page = 1
while True:
    tasks_res = session.get(f"{LS_URL}/api/tasks/?project={PID}&page={page}&page_size=100", timeout=15)
    if tasks_res.status_code != 200:
        if page == 1:
            print(f"❌ 獲取 Tasks 失敗: HTTP {tasks_res.status_code}")
            sys.exit(1)
        break
    
    tasks_data = tasks_res.json()
    page_tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else tasks_data
    if not page_tasks: break
    tasks.extend(page_tasks)
    page += 1

exported_count = 0

for task in tasks:
    if not task.get("is_labeled"): continue
    task_id = task.get("id")
    
    ann_res = session.get(f"{LS_URL}/api/tasks/{task_id}/annotations/", timeout=5)
    if ann_res.status_code != 200: continue
    
    annotations = ann_res.json()
    if not annotations:
        continue  # 沒有人工確認的標註
    
    latest_annotation = annotations[-1]
    results = latest_annotation.get("result", [])
    if not results:
        continue
    
    file_upload = task.get("file_upload") or task.get("data", {}).get("image", "")
    if not file_upload: continue
    filename = os.path.basename(file_upload).split("?")[0]
    
    yolo_lines = []
    
    for res in results:
        if res.get("type") == "rectanglelabels":
            labels = res["value"].get("rectanglelabels", [])
            if not labels: continue
            label_name = labels[0]
            
            yolo_cls_id = ENV_LABEL_TO_YOLO.get(label_name, 3)
            
            ls_x = res["value"]["x"]
            ls_y = res["value"]["y"]
            ls_w = res["value"]["width"]
            ls_h = res["value"]["height"]
            
            # Label Studio x,y 是左上角，轉換為中心點並標準化 0-1
            cx = (ls_x + ls_w / 2.0) / 100.0
            cy = (ls_y + ls_h / 2.0) / 100.0
            w = ls_w / 100.0
            h = ls_h / 100.0
            
            yolo_lines.append(f"{yolo_cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
    if yolo_lines:
        stem = Path(filename).stem
        txt_path = LABELS_DIR / f"{stem}.txt"
        txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        
        src_img = SOURCE_IMAGES_DIR / filename
        dst_img = IMAGES_DIR / filename
        if src_img.exists():
            shutil.copy(src_img, dst_img)
        
        exported_count += 1

print(f"✅ 成功下載並轉換 {exported_count} 筆人工標註至 DETR (YOLO BBox) 格式！")
