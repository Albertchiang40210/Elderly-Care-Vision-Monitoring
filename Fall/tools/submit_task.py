import os
import sys
from clearml import Task

# 1. 物理移除 AWS 髒憑證防禦
for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"]:
    if key in os.environ: del os.environ[key]

# 2. 憑證注入函數
def load_env_credentials():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip()

# 🎯 🌟 關鍵修正：宣告 project_name 參數，預設為 None
def main(project_name=None):
    load_env_credentials()
    
    # 🎯 鎖定：禁止所有 ClearML Agent 自帶的 Git/Auto 安裝干擾
    os.environ["CLEARML_DISABLE_GIT_DETECTION"] = "1"
    os.environ["CLEARML_AUTO_DIST"] = "0"
    
    # 若沒有從 Python 函式直接傳入 project_name，才解析 CLI 參數
    if not project_name:
        import argparse
        parser = argparse.ArgumentParser(description="ClearML 重訓點火器")
        parser.add_argument("--project", type=str, default="Fall_Detection", help="目標重訓專案名稱")
        # 避免在已被 FastAPI/Uvicorn 載入時因 CLI args 解析衝突而報錯
        args, _ = parser.parse_known_args()
        project_name = args.project

    print(f"[*] 🚀 [submit_task] 接收到 ClearML 重訓點火請求，目標專案: '{project_name}'")

    if project_name == "Hazard_Detection":
        task_name = "RTDETR_Cloud_Incremental_Training_Automated"
        project_id = 2
    elif "Action" in project_name or "Video" in project_name:
        task_name = "ActionTransformer_Cloud_Incremental_Training_Automated"
        project_id = 5
    else:
        task_name = "YOLOPose_Cloud_Incremental_Training_Automated"
        project_id = 1

    # 🎯 檢查 Label Studio 中對應 Project 已完成標註 (已按下 Submit) 的 Task 總數
    ls_url = os.getenv("LS_URL", "http://localhost:8082")
    labeled_count = 0

    try:
        import requests
        s = requests.Session()
        login_url = f"{ls_url}/user/login/"
        username = os.getenv("LABEL_STUDIO_USERNAME", "wang4021096@gmail.com")
        password = os.getenv("LABEL_STUDIO_PASSWORD", "")
        
        # 模擬登入獲取 CSRF Token & Session
        s.get(login_url, timeout=5)
        s.post(login_url, data={"email": username, "password": password, "csrfmiddlewaretoken": s.cookies.get("csrftoken", "")}, timeout=5)
        s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken", "")})
        
        res = s.get(f"{ls_url}/api/projects/{project_id}/tasks/", params={"page_size": 1000}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            tasks = data if isinstance(data, list) else data.get("tasks", data.get("results", []))
            labeled_count = sum(1 for t in tasks if t.get("is_labeled") or t.get("total_annotations", 0) > 0)
            print(f"📊 [Label Studio API 查核] 專案 '{project_name}' (Project ID: {project_id}) 目前已有 {labeled_count} 張標註完成的照片。")
        else:
            # 若專案 ID 查不到，嘗試透過專案列表 API 動態搜尋 ID
            projects_res = s.get(f"{ls_url}/api/projects/", timeout=5)
            if projects_res.status_code == 200:
                p_list = projects_res.json().get("results", [])
                for p in p_list:
                    if p.get("title") == project_name:
                        real_id = p.get("id")
                        tasks_res = s.get(f"{ls_url}/api/projects/{real_id}/tasks/", params={"page_size": 1000}, timeout=5)
                        if tasks_res.status_code == 200:
                            t_data = tasks_res.json()
                            tasks = t_data if isinstance(t_data, list) else t_data.get("tasks", t_data.get("results", []))
                            labeled_count = sum(1 for t in tasks if t.get("is_labeled") or t.get("total_annotations", 0) > 0)
                            print(f"📊 [動態 ID 查核] 專案 '{project_name}' (ID: {real_id}) 目前已有 {labeled_count} 張標註完成的照片。")
                        break
    except Exception as api_err:
        print(f"⚠️ 查詢 Label Studio (Project ID: {project_id}) 已標註數量失敗: {api_err}")

    # 🎯 門檻檢查邏輯
    if labeled_count < 10:
        print(f"ℹ️ [MLOps 門檻保護] 專案 '{project_name}' 在 Label Studio 中已 Submit 標註的照片為 {labeled_count} 張 (未達 10 張 Submit 門檻)，暫不觸發 ClearML 重訓。")
        return
    else:
        print(f"🔥 [MLOps 門檻達成] 專案 '{project_name}' 已累積 {labeled_count} 張 Submit 標註照片 (已達 10 張門檻)，正式點火發射 ClearML 重訓 Task！")

    # 檢查是否已有任務鎖定 (執行中或排隊中)
    active_tasks = Task.get_tasks(
        project_name=project_name,
        task_name=task_name,
        task_filter={"status": ["in_progress", "queued"]}
    )
    if active_tasks:
        print(f"🛑 專案 '{project_name}' 已有任務 {active_tasks[0].id} 執行中/排隊中，跳過重複點火。")
        return

    # 3. 準備執行腳本 (處理融合安裝邏輯)
    if "Action" in project_name or "Video" in project_name:
        script_name = "clearml_action_train_pipeline.py"
    elif "Fall" in project_name:
        script_name = "clearml_pose_train_pipeline.py"
    else:
        script_name = "clearml_train_pipeline.py"
    orig_pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    temp_pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".temp_pipeline.py")

    if not os.path.exists(orig_pipeline_path):
        print(f"❌ 找不到訓練腳本: {orig_pipeline_path}，無法建立 ClearML 任務！")
        return

    with open(orig_pipeline_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 🎯 🌟 [穩定性修正] 強制寫入：若缺少必要套件則自動安裝 (移除 S3 依賴，適應全地端架構)
    auto_install = (
        "import sys, subprocess, os\n"
        "try:\n"
        "    import ultralytics, clearml, pandas\n"
        "except ImportError:\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'clearml', 'pandas'])\n\n"
    )
    with open(temp_pipeline_path, "w", encoding="utf-8") as f:
        f.write(auto_install + code)

    # 4. 🎯 初始化 Task，並直接傳入 script 參數與停用 Git 偵測
    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        task_type="training",
        script=temp_pipeline_path,
        detect_repository=False
    )
    
    # 任務建立後可刪除暫存檔保持資料夾乾淨 (❌ 這裡不能刪！否則地端 Agent baremetal 執行會找不到檔案而秒掛)
    # try:
    #     os.remove(temp_pipeline_path)
    # except:
    #     pass

    # 🎯 🌟 [極度關鍵] 這裡必須明確告訴 Agent 什麼都不用做，直接跑腳本！
    task.set_base_docker(None)
    task.set_packages([]) # ❌ 強制禁用自動套件安裝，改由手動入口安裝
    task.output_uri = True

    print(f"📦 任務建立成功！Task ID: {task.id}")

    # 5. 排隊
    Task.enqueue(task=task, queue_name="default")
    print(f"✅ [發射成功] 專案 '{project_name}' 任務 {task.id} 順利進入 ClearML 'default' 隊列。")

if __name__ == "__main__":
    main()


