import os
import sys
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# =========================================================================
# 1. 讀取環境變數
# =========================================================================
def load_dotenv(path: Path) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

env_file_path = BASE_DIR / ".env"
if env_file_path.exists(): load_dotenv(env_file_path)
else: load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8080")
USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")
PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "")

def fail(msg: str) -> None:
    print(f"\n[X] {msg}")
    sys.exit(1)

# =========================================================================
# 2. 登入 Label Studio
# =========================================================================
print(f"[*] 正在連線並登入 Label Studio ({LS_URL}) ...")
session = requests.Session()
login_page_url = f"{LS_URL}/user/login/"
try:
    init_res = session.get(login_page_url, timeout=5)
    csrftoken = session.cookies.get('csrftoken', '')
except Exception as e: fail(f"無法連線至 Label Studio: {e}")

login_data = {"email": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken}
session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": login_page_url})
login_res = session.post(login_page_url, data=login_data, allow_redirects=True)
if "login" in login_res.url: fail("登入失敗，請檢查帳號密碼！")
print("✅ 登入成功！")

# =========================================================================
# 3. 建立專屬影片動作識別專案
# =========================================================================
PROJECT_NAME = "Action_Recognition_Video_V2"
PROJECT_NAME = "Action_Recognition_Video_V3"

# 檢查專案是否已存在
projects_res = session.get(f"{LS_URL}/api/projects/")
if projects_res.status_code == 200:
    for p in projects_res.json().get("results", []):
        if p.get("title") == PROJECT_NAME:
            fail(f"專案 '{PROJECT_NAME}' 已經存在！請直接至網頁端查看。")

# Label Studio 影片標註專用 UI (XML)
# 支援 Video 播放，以及給定時間段加上 Action 標籤 (Labels)
LABEL_CONFIG = """
<View>
  <Header value="Video Action Recognition (Whole Video Classification)" />
  <Video name="video" value="$video" />
  <Choices name="action" toName="video" showInline="true">
    <Choice value="fall" background="#FF0000"/>
    <Choice value="normal" background="#00FF00"/>
    <Choice value="kneel" background="#FFA500"/>
    <Choice value="sitdown" background="#FFFF00"/>
    <Choice value="walk" background="#0000FF"/>
  </Choices>
</View>
"""

print(f"[*] 正在建立新專案: {PROJECT_NAME} ...")
create_res = session.post(
    f"{LS_URL}/api/projects/",
    json={"title": PROJECT_NAME, "label_config": LABEL_CONFIG.strip()}
)

if create_res.status_code not in [200, 201]:
    fail(f"建立專案失敗: {create_res.text}")

project_id = create_res.json().get("id")
print(f"✅ 專案建立成功！專案 ID: {project_id}")

# =========================================================================
# 4. 連接 Local Storage (指向刚建立的 label_studio_data/videos 資料夾)
# =========================================================================
print(f"[*] 正在設定 Local Storage (影片資料來源) ...")
# 取得 Docker 容器內的路徑
videos_dir = "/label-studio/data/label_studio_data/videos"

storage_payload = {
    "project": project_id,
    "path": videos_dir,
    "title": "Local_Video_Storage",
    "use_blob_urls": True
}

storage_res = session.post(f"{LS_URL}/api/storages/localfiles", json=storage_payload)
if storage_res.status_code in [200, 201]:
    storage_id = storage_res.json().get("id")
    print("✅ Local Storage 設定成功！")
    
    # 觸發一次 Sync 同步影片
    print("[*] 正在同步資料夾內的影片至 Label Studio...")
    session.post(f"{LS_URL}/api/storages/localfiles/{storage_id}/sync")
    print("✅ 同步完成！您剛才放入的影片應該已經出現在專案的 Tasks 裡面了。")
else:
    print(f"⚠️ Local Storage 設定失敗，您可能需要稍後手動至網頁端設定。錯誤: {storage_res.text}")

print(f"\n🎉 全部完成！請前往 {LS_URL} 查看名為 '{PROJECT_NAME}' 的新專案！")
