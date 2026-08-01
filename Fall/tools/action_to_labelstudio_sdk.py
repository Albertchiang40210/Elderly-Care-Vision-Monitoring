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
# print(f"[*] 正在建立 Session 並嘗試登入 {LS_URL} ...")
# session = requests.Session()
# login_page_url = f"{LS_URL}/user/login/"
# try:
#     init_res = session.get(login_page_url, timeout=5)
#     csrftoken = session.cookies.get('csrftoken', '')
# except Exception as e: fail(f"無法連線至 Label Studio 服務: {e}")

# login_data = {"email": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken}
# session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": login_page_url})
# login_res = session.post(login_page_url, data=login_data, allow_redirects=True)
# if "login" in login_res.url: fail("帳號或密碼錯誤，請檢查 USERNAME 和 PASSWORD 設定！")
# print("🎉 [登入成功] 已獲取合法網頁 Session 憑證！")
print("⚠️ [測試模式] 已跳過 Label Studio 登入，直接執行本地影片推論...")


# =========================================================================
# 3. 讀取 ActionTracker 並執行影片/時間序列推論
# =========================================================================
print(f"🔄 正在載入 ActionTracker (YOLO-Pose + Tracker)...")
tracker = ActionTracker(pose_model_path=DEFAULT_MODEL_PATH, sequence_length=30)

# TODO: 這裡示範如何對一個「影片」進行逐影格讀取並推論
# 實務上這會從 Label Studio 抓取 Video 類型的 Task URL
video_path = str(PROJECT_ROOT / "media" / "videos" / "fallforward_1P.mp4")
if not os.path.exists(video_path):
    print(f"⚠️ 找不到測試影片 {video_path}，腳本執行結束。")
    print("💡 請確認影片已放置於 media/videos/ 資料夾內。")
    sys.exit(0)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

print("🎬 開始處理影片特徵收集...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 使用 tracker 追蹤並取得符合條件的序列
    results, ready_sequences = tracker.process_frame(frame, conf_thres=CONF_THRES)
    
    # 檢查是否有人達標 (收集滿 sequence_length 影格的特徵)
    for track_id, pose_seq in ready_sequences.items():
        action_label = tracker.predict_action(pose_seq)
        if action_label == "fall":
            print(f"🚨 [Frame {frame_idx}] 偵測到 ID: {track_id} 發生跌倒 (Fall)！")
            # 這裡可以實作呼叫 Label Studio API，推送一段包含 start_frame 和 end_frame 的 prediction
            
    frame_idx += 1

cap.release()
print("✅ 影片特徵收集與動作辨識模擬完成。")
