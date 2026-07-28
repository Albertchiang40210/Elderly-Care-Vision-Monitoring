import os
import sys
from pathlib import Path
import cv2
import requests
import json
import boto3
import numpy as np
import shutil
import time  
import base64  # 🎯 Base64 解碼模組
from clearml import Model  

# 🚨 核心大一統：全面換裝為與前線和 ClearML 後台重訓同構的 DEIM-DETR (RT-DETR)
from ultralytics import RTDETR

# =========================================================================
# 1. 參數與環境變數配置區 (🔑 核心防禦：修正為 100% 讀取 tools/.env 完全體)
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
    print(f"✅ [SDK 中繼站] 已成功物理強注環境變數: {env_file_path}")
else:
    load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8082") 
PROJECT_ID = int(os.getenv("LS_PROJECT_ID", "1").strip())  
CONF_THRES = float(os.getenv("CONF_THRES", "0.25")) 

# 🎯 【Label Studio 網頁登入帳密】
USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")  
PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "")     

# 🌟 模型路徑定義
ACTIVE_MODEL_PATH = str(PROJECT_ROOT / "models" / "active_rt_detr.pt")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "rtdetr-l.pt")

MODEL_PATH = ACTIVE_MODEL_PATH if os.path.exists(ACTIVE_MODEL_PATH) else DEFAULT_MODEL_PATH

IMAGES_DIR = PROJECT_ROOT / "active_learning_dataset" / "images"
LABELS_DIR = PROJECT_ROOT / "active_learning_dataset" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 🎯 安養中心 5 大純動態危險物品類別字典
ENVIRONMENT_COCO = {56: "obstacle", 59: "walker", 60: "walker"}
ENV_LABEL_TO_YOLO = {"wheelchair": 0, "slipper": 1, "wire": 2, "obstacle": 3, "walker": 4}



def fail(msg: str) -> None:
    print(f"\n[X] {msg}")
    sys.exit(1)


# =========================================================================
# 🔄 🎯 自動取得地端 ClearML 最新「最強大腦」的閉環控制
# =========================================================================
def get_latest_best_model_from_cloud():
    local_model_target = ACTIVE_MODEL_PATH
    os.makedirs(os.path.dirname(local_model_target), exist_ok=True)
    try:
        print("\n🔍 正在連線至 ClearML 尋找地端倉庫最新的 RT-DETR 'best' 模型...")

        models_found = Model.query_models(project_name="Fall_Detection", tags=["detr", "best"])

        if not models_found:
            print("ℹ️ ClearML 雲端目前尚未有任何帶有 'best' 標籤的模型上線。")
            raise FileNotFoundError("No best model on ClearML yet")
        models_found = sorted(models_found, key=lambda m: m.created, reverse=True)
        cloud_model = models_found[0]
        print(f"🎯 成功在雲端鎖定最新模型: {cloud_model.name} (創建時間: {cloud_model.created}, ID: {cloud_model.id})")
        downloaded_path = cloud_model.get_local_copy()
        if downloaded_path and os.path.exists(downloaded_path):
            print(f"📥 成功從 S3 下載最新模型！暫存路徑: {downloaded_path}")
            shutil.copy(downloaded_path, local_model_target)
            print(f"🔄 已成功使用雲端最新模型覆蓋本地端：{local_model_target}，準備進行更精準的推理！\n")
            global MODEL_PATH
            MODEL_PATH = local_model_target
            return True
    except Exception as e:
        print(f"ℹ️ 查詢或下載雲端模型失敗 (原因: {e})。")
        print(f"⚠️ 將採用備用方案：直接使用本地既有的 '{Path(MODEL_PATH).name}' 進行推理。\n")
        return False


# =========================================================================
# 2. 模擬瀏覽器登入
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
# 📡 自動探查專案標籤介面設定
# =========================================================================
FROM_NAME, TO_NAME = "label", "image"
PROJECT_TITLE = ""
try:
    proj_url = f"{LS_URL}/api/projects/{PROJECT_ID}/"
    session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
    p_res = session.get(proj_url, timeout=5)
    if p_res.status_code == 200:
        p_json = p_res.json()
        PROJECT_TITLE = p_json.get("title", "")
        label_config = p_json.get("label_config", "")
        import re
        from_match = re.search(r'name="([^"]+)"\s+toName=', label_config)
        to_match = re.search(r'toName="([^"]+)"', label_config)
        if from_match: FROM_NAME = from_match.group(1)
        if to_match: TO_NAME = to_match.group(1)
        print(f"📡 [介面探查成功] 動態綁定專案 '{PROJECT_TITLE}' 標籤映射: from_name='{FROM_NAME}', to_name='{TO_NAME}'")
except Exception as e: print(f"⚠️ 探查專案介面失敗，使用預設對齊 (label/image)。原因: {e}")


# =========================================================================
# 🔥 ✨ 【第十九關完全體】修正 400 參數並強制引爆 S3 同步與實體化落地
# =========================================================================
print("\n🌐 [核心第十九關] 正在引爆地端數據流，並強制實體化至專案任務列表...")

try:
    list_url = f"{LS_URL}/api/storages/s3?project={PROJECT_ID}"
    session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
    res = session.get(list_url, timeout=10)
    real_id = None
    if res.status_code == 200:
        storages = res.json()
        storages = storages.get("results", storages) if isinstance(storages, dict) else storages
        if isinstance(storages, list):
            for s in storages:
                if (s.get('title') == "S3_Import_Images" and s.get('project') == PROJECT_ID) or (s.get('project') == PROJECT_ID and real_id is None):
                    real_id = s['id']
        if real_id:
            print(f"🔥 [確定點火] 目標 S3 通道 ID: {real_id}")
            session.post(f"{LS_URL}/api/storages/s3/{real_id}/sync", timeout=15)
            session.post(f"{LS_URL}/api/storages/s3/{real_id}/realize", timeout=15)
            session.post(f"{LS_URL}/api/storages/s3/{real_id}/reimport", timeout=15)
except Exception as e: print(f"💥 [崩潰性異常] 第十九關腳本直接噴錯: {e}")


# =========================================================================
# 3. 智慧輪詢阻斷防線 (等候非同步資料庫寫入完畢)
# =========================================================================
tasks_url = f"{LS_URL}/api/tasks/"
session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
existing_tasks = []

print("\n⏳ [進入智慧阻斷防線] 開始輪詢 Label Studio 資料庫，等待非同步 Tasks 實體化...")
for i in range(10):
    tasks_res = session.get(tasks_url, params={"project": PROJECT_ID, "page_size": 1000}, timeout=10)
    if tasks_res.status_code != 200:
        tasks_res = session.get(f"{LS_URL}/api/projects/{PROJECT_ID}/tasks/", params={"page_size": 1000}, timeout=10)
    try:
        data = tasks_res.json()
        current_tasks = data.get("results", data.get("tasks", [])) if isinstance(data, dict) else data
    except Exception: current_tasks = []

    if isinstance(current_tasks, list) and len(current_tasks) > 0:
        existing_tasks = current_tasks
        print(f"🎉 [破防成功] 偵測到資料庫已實體化出 {len(existing_tasks)} 個任務！")
        break
    time.sleep(2)


# =========================================================================
# 4. 对全专案 task 跑 DEIM-DETR 推论 (強大可靠的單一任務迴圈注入管道)
# =========================================================================
pending = existing_tasks.get("results", existing_tasks.get("tasks", existing_tasks)) if isinstance(existing_tasks, dict) else existing_tasks
print(f"[*] 共 {len(pending)} 個 task 將使用 DEIM-DETR 進行環境智慧方形框標註")
if not pending: sys.exit(0)

get_latest_best_model_from_cloud()
model = RTDETR(MODEL_PATH)
pushed = 0

for idx, task in enumerate(pending, 1):
    if not isinstance(task, dict) or "data" not in task: continue
    task_image_url = task["data"].get("image", "")
    
    # Base64 智慧解碼路徑
    real_s3_url = None
    if "fileuri=" in task_image_url:
        try:
            b64_str = task_image_url.split("fileuri=")[-1]
            b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
            real_s3_url = base64.b64decode(b64_str).decode("utf-8", errors="ignore")
        except Exception: pass
    else: real_s3_url = task_image_url

    if not real_s3_url: real_s3_url = task_image_url
    filename = real_s3_url.split("/")[-1] if "/" in real_s3_url else f"task_{task['id']}.jpg"
    img_path = IMAGES_DIR / filename
    
    # ─── 業界標準增量快取設計 ───
    label_txt_path = LABELS_DIR / f"{img_path.stem}.txt"
    if img_path.exists() and label_txt_path.exists():
        print(f"⏭️  [{idx}/{len(pending)}] 檔案與標籤已存在本機快取，跳過處理: {filename}")
        pushed += 1
        continue
        
    print(f"📸 [{idx}/{len(pending)}] 正在對著目標影像發動 AI 智慧推理流程: {filename}")

    # 下載影像邏輯
    img_cv2 = None
    if not img_path.exists():
        if real_s3_url.startswith("s3://") or "s3.amazonaws.com" in real_s3_url:
            if "s3.amazonaws.com" in real_s3_url:
                parts = real_s3_url.replace("https://", "").replace("http://", "").split("/")
                bucket, key = parts[0].split(".")[0], "/".join(parts[1:])
            else:
                _path_parts = real_s3_url.replace("s3://", "").split("/", 1)
                bucket, key = _path_parts[0], _path_parts[1]
            try:
                s3_client = boto3.client('s3')
                try: s3_obj = s3_client.get_object(Bucket=bucket, Key=key)
                except s3_client.exceptions.NoSuchKey:
                    alt_key = key.replace("snapshots/", "") if key.startswith("snapshots/") else f"snapshots/{key}"
                    s3_obj = s3_client.get_object(Bucket=bucket, Key=alt_key)
                img_bytes = s3_obj['Body'].read()
                img_cv2 = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(str(img_path), img_cv2)
            except Exception: continue
    else: img_cv2 = cv2.imread(str(img_path))

    if img_cv2 is None: continue
    img_h, img_w, _ = img_cv2.shape

    # ─── 🛡️ 智慧標註分支：偵測是否已有人工標註 (Option B) ───
    # 100% 確保準確，直接向 Label Studio API 請求該 Task 當前的完整明細（包含 annotations 列表）
    annotations = []
    try:
        session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
        detail_res = session.get(f"{LS_URL}/api/tasks/{task['id']}/", timeout=5)
        if detail_res.status_code == 200:
            annotations = detail_res.json().get("annotations", [])
    except Exception as ex:
        print(f"  [!] 獲取 Task #{task['id']} 詳情失敗: {ex}")
            
    if annotations:
        print(f"📥 [{idx}/{len(pending)}] 任務 ID {task['id']} 已有標記，正在從網頁下載人工/最新標註座標...")
        latest_anno = annotations[-1]  # 取得最新的一筆標記
        yolo_lines = []
        result = latest_anno.get("result", [])
        for item in result:
            if item.get("type") != "rectanglelabels": continue
            val = item.get("value", {})
            labels = val.get("rectanglelabels", [])
            if not labels: continue
            label_name = labels[0]
            if label_name not in ENV_LABEL_TO_YOLO: continue
            cls_id = ENV_LABEL_TO_YOLO[label_name]
            
            x = val.get("x", 0.0)
            y = val.get("y", 0.0)
            w = val.get("width", 0.0)
            h = val.get("height", 0.0)
            
            yolo_x = (x + w / 2.0) / 100.0
            yolo_y = (y + h / 2.0) / 100.0
            yolo_w = w / 100.0
            yolo_h = h / 100.0
            
            yolo_x = min(max(yolo_x, 0.0), 1.0)
            yolo_y = min(max(yolo_y, 0.0), 1.0)
            yolo_w = min(max(yolo_w, 0.0), 1.0)
            yolo_h = min(max(yolo_h, 0.0), 1.0)
            
            yolo_lines.append(f"{cls_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")
            
        # 寫入本地 labels
        (LABELS_DIR / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        print(f"  -> 💾 已成功寫入/保留人工標註至本地: {img_path.stem}.txt (共 {len(yolo_lines)} 個方框)")
        pushed += 1
        continue

    # ─── 🛡️ 智慧標註分支：依據專案類型進行精準篩選 ───
    # 判斷專案名稱是否包含 Fall/跌倒，若有則只打 person 框；否則按危險雜物打標
    IS_FALL_PROJECT = "fall" in PROJECT_TITLE.lower() or "跌倒" in PROJECT_TITLE
    
    results = model.predict(img_cv2, conf=CONF_THRES, verbose=False)
    ls_result, yolo_lines = [], []
    
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            
            if IS_FALL_PROJECT:
                # 跌倒專案：只畫 person (COCO 類別 0 為 person)
                if cls_id != 0: continue
                label_name = "person"
                yolo_cls_id = 0
            else:
                # 危險雜物專案：只畫 5 大危險雜物
                if cls_id not in ENVIRONMENT_COCO: continue
                label_name = ENVIRONMENT_COCO[cls_id]
                yolo_cls_id = ENV_LABEL_TO_YOLO.get(label_name, 0)
            
            conf = float(box.conf[0].item())
            xyxy = box.xyxy.cpu().numpy()[0]
            
            ls_x = float(round((xyxy[0] / img_w) * 100, 4))
            ls_y = float(round((xyxy[1] / img_h) * 100, 4))
            ls_w = float(round(((xyxy[2] - xyxy[0]) / img_w) * 100, 4))
            ls_h = float(round(((xyxy[3] - xyxy[1]) / img_h) * 100, 4))
            
            ls_result.append({
                "from_name": FROM_NAME, "to_name": TO_NAME, "type": "rectanglelabels",
                "value": {"x": ls_x, "y": ls_y, "width": ls_w, "height": ls_h, "rectanglelabels": [label_name]},
                "score": conf
            })
            yolo_lines.append(f"{yolo_cls_id} {box.xywhn.cpu().numpy()[0][0]:.6f} {box.xywhn.cpu().numpy()[0][1]:.6f} {box.xywhn.cpu().numpy()[0][2]:.6f} {box.xywhn.cpu().numpy()[0][3]:.6f}")

    if not ls_result:
        ls_result.append({
            "from_name": FROM_NAME, "to_name": TO_NAME, "type": "rectanglelabels",
            "value": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "rectanglelabels": ["chair"]}, "score": 1.0
        })
        yolo_lines.append("1 0.005000 0.005000 0.010000 0.010000")

    (LABELS_DIR / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    # =========================================================================
    # 🎯 🌟【半自動 / AI 預標註模式】：清除舊預測並注入 DEIM-DETR 預測 (Predictions)
    # =========================================================================
    session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
    try:
        # 清除此任務當前舊有的預測標註 (Predictions)
        detail_res = session.get(f"{LS_URL}/api/tasks/{task['id']}/", timeout=5)
        if detail_res.status_code == 200:
            latest_preds = detail_res.json().get("predictions", [])
            if latest_preds:
                for old_pred in latest_preds:
                    old_id = old_pred.get("id")
                    if old_id:
                        session.delete(f"{LS_URL}/api/predictions/{old_id}/", timeout=5)
                print(f"  -> 🧹 [預測查殺] 已清除舊有的 {len(latest_preds)} 個過期預測標註")
    except Exception as ex:
        print(f"  [!] 清除舊預測標註異常: {ex}")

    # 🚀 物理注入 AI 預標註 (Predictions - 人工審核半自動模式)
    pred_url = f"{LS_URL}/api/predictions/"
    avg_score = float(np.mean([item.get("score", 1.0) for item in ls_result])) if ls_result else 1.0
    pred_payload = {
        "task": task["id"],
        "model_version": "DEIM-DETR",
        "result": ls_result,
        "score": avg_score
    }
    res_pred = session.post(pred_url, json=pred_payload, timeout=30)
    
    try:
        if res_pred.status_code in [200, 201]:
            pushed += 1
            print(f"  -> 🎯 [半自動預標註成功] 已注入 AI 預測框 (Prediction)！等待人工至 Label Studio 網頁開箱審核與 Submit (狀態碼: {res_pred.status_code})")
        else:
            print(f"  -> ❌ 預標註注入失敗！狀態碼: {res_pred.status_code} | 原因: {res_pred.text[:200]}")
    except Exception as e:
        print(f"  -> ❌ 通訊異常: {e}")

print(f"\n[OK] 大功告成！共完成 {pushed} 個專案任務的 DEIM-DETR AI 半自動預標註 (Predictions)。")
print("🔥 完美切換至「半自動人工開箱審核模式」，人工 Submit 後點火飛輪 Webhook 將自動觸發！")


#「它是後台的『 AI 預標註小幫手（半自動輔助）』，負責把跌倒照片自動預先畫好框 (Predictions)，等待人工在網頁點擊 Submit 審核標註。」
#這個檔案是我們主動學習數據飛輪的靈魂（半自動人機協同模式）：
#1. 模擬人類登入網頁：
#   讀取 .env 帳密，用 requests.Session 登入 Label Studio 標註平台。
#2. 智慧防禦與 S3 下沉對齊：
#   發現雲端 S3 照片時，透過 Boto3 串流下載回地端實體目錄，確保標註與 ClearML 重訓資料夾完整。
#3. DEIM-DETR 智慧環境預標註 (Predictions)：
#   利用 DEIM-DETR 辨識關鍵物體 (person, chair, sofa, bed, tv)，並將預測結果作為 Predictions 注入 Label Studio。
#4. 人工開箱審核與點火 (Semi-Automatic Workflow)：
#   標註任務保持待審核狀態，標註員登入 Label Studio 後可直接看到 AI 畫好的預測框，經確認/微調後點擊 Submit 提交，進而觸發 Webhook 引爆背景 ClearML 重訓飛輪。