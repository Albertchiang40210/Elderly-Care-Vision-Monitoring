import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

env_file_path = BASE_DIR / ".env"
if env_file_path.exists(): load_dotenv(env_file_path)
else: load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8082")
USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")
PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "")
PROJECT_ID = 5 # 這是我們剛剛建立的 Action_Recognition_Video_V3 專案 ID

def main():
    print(f"[*] 正在連線至 Label Studio ({LS_URL})...")
    session = requests.Session()
    login_page_url = f"{LS_URL}/user/login/"
    try:
        init_res = session.get(login_page_url, timeout=5)
        csrftoken = session.cookies.get('csrftoken', '')
    except Exception as e:
        print(f"❌ 無法連線至 Label Studio: {e}")
        sys.exit(1)
        
    login_data = {"email": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken}
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": login_page_url})
    login_res = session.post(login_page_url, data=login_data, allow_redirects=True)
    if "login" in login_res.url:
        print("❌ 登入失敗！請檢查帳號密碼")
        sys.exit(1)
        
    print("[*] 登入成功！正在下載最新標註結果...")
    
    export_url = f"{LS_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON"
    res = session.get(export_url)
    if res.status_code == 200:
        data = res.json()
        
        # 解析標註結果
        annotations = {}
        for task in data:
            if not task.get("annotations"): continue
            
            # 從 Label Studio 的檔案路徑提取原本的檔名
            # 例如: /data/local-files/?d=backend/static/images/fallforward_1P.mp4
            file_url = task["data"]["video"]
            filename = os.path.basename(file_url.split("?d=")[-1] if "?d=" in file_url else file_url)
            
            # 取得最新的一筆標註
            latest_anno = task["annotations"][-1]
            if not latest_anno.get("result"): continue
            
            # 在 Choices 介面中，結果通常是 ["fall"]
            choices = latest_anno["result"][0]["value"].get("choices")
            if choices:
                annotations[filename] = choices[0]
                
        out_path = PROJECT_ROOT / "active_learning_dataset" / "annotations.json"
        with open(out_path, "w") as f:
            json.dump(annotations, f, indent=2)
            
        print(f"✅ 成功下載 {len(annotations)} 筆人工標註！已儲存至: {out_path}")
    else:
        print(f"❌ 無法下載標註，狀態碼: {res.status_code}")

if __name__ == "__main__":
    main()
