import sys, subprocess, os
try:
    import ultralytics, clearml, boto3
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'clearml', 'boto3'])

# 載入本機 AWS 憑證環境變數與 ClearML S3 配置覆蓋，以防 S3 寫入權限錯誤
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(CURRENT_DIR, '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k.strip()] = v.strip()
    os.environ['CLEARML_SDK__AWS__S3__KEY'] = os.environ.get('AWS_ACCESS_KEY_ID', '')
    os.environ['CLEARML_SDK__AWS__S3__SECRET'] = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
    os.environ['CLEARML_SDK__AWS__S3__USE_CREDENTIALS_CHAIN'] = 'true'

import os
from clearml import Task, OutputModel
from ultralytics import YOLO, RTDETR

def main():
    # 1. 初始化 Task
    task = Task.init(
        project_name=os.environ.get("CLEARML_PROJECT", "Fall_Detection"), 
        task_name=os.environ.get("CLEARML_TASK_NAME", "Auto_Incremental_Training")
    )
    
    project_name = task.get_project_name()
    IS_FALL_PROJECT = "fall" in project_name.lower() or "跌倒" in project_name
    
    task.output_uri = f"s3://aipe03-3/clearml-artifacts/models/{project_name.lower()}/"
    print(f"====== 🍏 [Agent 遠端] 背景運算節點已成功接單 (專案: {project_name}) ======")
    
    if IS_FALL_PROJECT:
        model_type = "yolo_pose"
        default_model_name = "yolo11s-pose.pt"
        tag_list = ["yolo_pose", "best"]
    else:
        model_type = "detr"
        default_model_name = "rtdetr-l.pt"
        tag_list = ["detr", "best"]

    base_model_path = default_model_name
    try:
        from clearml import Model
        print(f"🔍 [增量鏈結] 正在檢查雲端是否有上一輪產出的 {model_type} 最強大腦...")
        cloud_bests = Model.query_models(project_name=project_name, tags=tag_list)
        if cloud_bests:
            cloud_bests = sorted(cloud_bests, key=lambda m: m.created, reverse=True)
            latest_cloud_model = cloud_bests[0]
            print(f"📥 [找到大腦] 發現上一輪的最新 {model_type} 模型 (ID: {latest_cloud_model.id})，正在拉取權重...")
            downloaded_base = latest_cloud_model.get_local_copy()
            if downloaded_base and os.path.exists(downloaded_base):
                base_model_path = downloaded_base
                print("🔄 [繼承成功] 成功載入最新雲端權重，模型將在此基礎上增量微調！")
        else:
            print(f"ℹ️ 雲端尚未有任何 {tag_list} 模型，將從原始 '{default_model_name}' 開始冷啟動。")
    except Exception as e:
        print(f"⚠️ 嘗試拉取雲端模型失敗 ({e})，降級使用 '{default_model_name}'。")
    
    if IS_FALL_PROJECT:
        model = YOLO(base_model_path)
    else:
        model = RTDETR(base_model_path)
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(CURRENT_DIR, 'data.yaml')

    # 3. 開始訓練
    # 🎯 直接在 train 內加入 plots=False，這能 100% 關閉大圖生成與上傳，
    # 同時完美避開了 ImportError 版本相容問題，並大幅節省連線頻寬！
    model.train(
        data=data_yaml_path, 
        epochs=1, 
        imgsz=640, 
        batch=4, 
        device='cpu',  # 測試用 CPU 跑 1 epoch 即可
        plots=False    # 阻擋生成/上傳大圖，確保連線不中斷
    )

    # =========================================================================
    # 🎯 動態綁定並強制為 ClearML 自動偵測到的模型貼上 'detr' 與 'best' 標籤
    # =========================================================================
    # 直接向當前的 task 索取它在訓練過程中自動偵測並準備上傳的所有模型物件
    models = task.get_models()
    
    if models and 'output' in models and len(models['output']) > 0:
        print("\n🚀 成功偵測到 ClearML 已自動捕捉到訓練產出的模型！")
        # 取得最新產出的輸出模型物件（最後一個元素）
        output_model = models['output'][-1]
        
        # 強制為這個 S3 上的模型貼上 'detr' 與 'best' 標籤，讓自動更新 SDK 可以一秒識別並下載
        output_model.tags = ['detr', 'best']
        print("✅ [S3 同步完成] 最新權重已成功上傳至 S3，並標記為 'detr', 'best' 標籤！\n")
    else:
        # 💡 Fallback 安全備份機制：如果 ClearML 沒有在第一時間自動捕捉，我們再用手動方式去抓
        print("ℹ️ ClearML 未能自動捕捉模型，啟用 Fallback 機制尋找本地檔案...")
        local_best_path = "runs/detect/train/weights/best.pt"
        
        if os.path.exists(local_best_path):
            # 🎯 雙重保險：手動建立 OutputModel 時，強制宣告目的地為 S3 儲存桶
            output_model = OutputModel(
                task=task, 
                name="RTDETR_Cloud_Incremental_Training_Automated",
                destination="s3://aipe03-3/clearml-artifacts/models/fall_detection/"
            )
            output_model.update_weights(weights_filename=local_best_path, auto_delete_local_copy=False)
            
            # 🎯 語法修正：改為屬性賦值，避免報錯
            output_model.tags = ['detr', 'best']
            print("✅ [Fallback 同步完成] 已手動將本地模型推送至 S3 並標記 'detr', 'best'！\n")

        else:
            print("\n⚠️ 警告：找不到任何自動或本地的模型權重，請檢查訓練是否正常結束。\n")

if __name__ == "__main__":
    main()


#「它是我們重訓飛輪的『發動機』，讓模型可以在後台『自動進修』，並把新學到的知識存在雲端。」
#這個檔案就是你這套 Active Learning 系統中最硬核、最有價值的 MLOps 雲端重訓管線（Pipeline）。它的運作流程可以用三個步驟白話解釋：
#雲端掛號註冊（ClearML Task）：
#程式啟動後，會先跟 ClearML 平台註冊一個叫做 "Fall_Detection" 的任務，告訴平台：「我要開始訓練 RT-DETR 了！」同時，它還指定了 s3://aipe03-3/clearml-artifacts/models/fall_detection/ 作為雲端倉庫，只要模型一訓練完，立刻自動把最新的模型權重（.pt）備份到 AWS S3！
#地端特工接單（ClearML Agent）：
#這段程式平常不會在你本機直接消耗大量資源。它是被送到 ClearML 的隊列（Queue）中，等你左邊視窗的 Agent（運算節點）看到有新訂單時，把任務接過去，在背景悄悄執行。
#RT-DETR 知識升級（Incremental Training）：
#Agent 接單後，會自動加載你的 rtdetr-l.pt 權重檔，並讀取你指定的絕對路徑 data.yaml，用新進來的資料進行微調訓練（雖然 Demo 只跑 1 epoch，但這已經完整實現了 MLOps 閉環！）。