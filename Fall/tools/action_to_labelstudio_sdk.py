import os
import sys
from pathlib import Path
import cv2
import requests
import json
import time

# 確保能 import modules 內的 ActionTracker
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from modules.action_tracker import ActionTracker

# =========================================================================
# 1. 參數與環境變數配置區
# =========================================================================
def load_dotenv(path: Path) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

env_file_path = BASE_DIR / ".env"
if env_file_path.exists():
    load_dotenv(env_file_path)
else:
    load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8080") 
USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")  
PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "")     

DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "yolo11s-pose.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))

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
session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": login_page_url})
login_res = session.post(login_page_url, data=login_data, allow_redirects=True)
if "login" in login_res.url: fail("帳號或密碼錯誤，請檢查 USERNAME 和 PASSWORD 設定！")
print("🎉 [登入成功] 已獲取合法網頁 Session 憑證！")
session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})

# =========================================================================
# 3. 尋找 Action Recognition 專案並同步 Local Storage
# =========================================================================
print("🔍 正在搜尋 Action Recognition 影片標註專案...")
projects_res = session.get(f"{LS_URL}/api/projects/", timeout=5)
project_id = None
if projects_res.status_code == 200:
    for p in projects_res.json().get("results", []):
        if p.get("title", "").startswith("Action_Recognition_Video"):
            project_id = p.get("id")
            print(f"✅ 找到專案: {p.get('title')} (ID: {project_id})")
            break

if not project_id:
    fail("找不到開頭為 Action_Recognition_Video 的專案，請先確認專案已建立。")

# 同步 Local Storage
print("[*] 嘗試同步 Local Storage...")
storage_url = f"{LS_URL}/api/storages/localfiles/?project={project_id}"
storage_res = session.get(storage_url, timeout=5)
if storage_res.status_code == 200:
    for st in storage_res.json():
        st_id = st.get("id")
        if st_id: session.post(f"{LS_URL}/api/storages/localfiles/{st_id}/sync", timeout=5)
time.sleep(1)

# =========================================================================
# 4. 取得待標註的 Tasks 並進行 AI 預測
# =========================================================================
tasks = []
page = 1
while True:
    tasks_res = session.get(f"{LS_URL}/api/projects/{project_id}/tasks/", params={"page": page, "page_size": 100}, timeout=15)
    if tasks_res.status_code != 200:
        if page == 1: print(f"⚠️ 無法取得 Tasks，狀態碼: {tasks_res.status_code}, 內容: {tasks_res.text[:200]}")
        break
    
    tasks_data = tasks_res.json()
    page_tasks = tasks_data.get("results", []) if isinstance(tasks_data, dict) else tasks_data
    if not page_tasks: break
    tasks.extend(page_tasks)
    page += 1

if not tasks: fail("專案中沒有任何影片 Task！(或撈取失敗)")

print(f"🔥 共找到 {len(tasks)} 個影片 Task，準備進行 AI 預測...")

print(f"🔄 正在載入 ActionTracker (YOLO-Pose + Tracker)...")
tracker = ActionTracker(pose_model_path=DEFAULT_MODEL_PATH, sequence_length=30)

videos_dir = PROJECT_ROOT / "label_studio_data" / "videos"
total_pushed = 0

for idx, task in enumerate(tasks, 1):
    task_id = task["id"]
    video_url = task.get("data", {}).get("video", "")
    import urllib.parse
    raw_path = video_url.split("=")[-1] if "=" in video_url else video_url
    filename = Path(urllib.parse.unquote(raw_path)).name
    video_path = videos_dir / filename
    
    if not video_path.exists():
        print(f"⚠️ 找不到本地影片: {video_path}")
        continue
        
    print(f"\n🎬 [Task {task_id}] 正在處理影片 ({idx}/{len(tasks)}): {filename} ...")
    cap = cv2.VideoCapture(str(video_path))
    
    action_found = "normal"  # 預設為 normal
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results, ready_sequences = tracker.process_frame(frame, conf_thres=CONF_THRES)
        for track_id, pose_seq in ready_sequences.items():
            action_label = tracker.predict_action(pose_seq)
            if action_label == "fall":
                action_found = "fall"
                break
        
        if action_found == "fall":
            break # 提早結束，判定為跌倒
            
    cap.release()
    print(f"🎯 AI 預測結果: {action_found.upper()}")
    
    # 建立 Prediction Payload (整支影片分類)
    pred_payload = {
        "task": task_id,
        "model_version": "ActionTracker-AutoLabel",
        "result": [
            {
                "from_name": "action",
                "to_name": "video",
                "type": "choices",
                "value": {
                    "choices": [action_found]
                }
            }
        ],
        "score": 0.95
    }
    
    # 先清除舊有過期預測
    detail_res = session.get(f"{LS_URL}/api/tasks/{task_id}/", timeout=5)
    if detail_res.status_code == 200:
        for old_pred in detail_res.json().get("predictions", []):
            if old_pred.get("id"):
                session.delete(f"{LS_URL}/api/predictions/{old_pred['id']}/", timeout=5)

    res_pred = session.post(f"{LS_URL}/api/predictions/", json=pred_payload, timeout=10)
    if res_pred.status_code in [200, 201]:
        total_pushed += 1
        print(f"✅ 成功寫入 Prediction！")
    else:
        print(f"❌ 寫入 Prediction 失敗: {res_pred.text}")

print(f"\n🎉 大功告成！共為 {total_pushed} 支影片完成了 AI 預標註！")
print("👉 請前往 Label Studio 重新整理網頁查看成果，直接審核即可！")
