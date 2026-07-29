import os
from clearml import Task, OutputModel
from ultralytics import YOLO

def main():
    # 1. 初始化 Task
    # 這是專為「人體姿態/跌倒辨識」打造的專屬高速公路！
    task = Task.init(
        project_name="Fall_Pose_Detection", 
        task_name="YOLO_Pose_Incremental_Training_Automated"
    )
    
    # 🎯 🌟 [地端隔離分流] 依據專案類型將 YOLO-Pose 與 RT-DETR 模型分開儲存
    p_name = task.get_project_name() or "Fall_Pose_Detection"
    model_subfolder = "yolo_pose"
    
    local_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "active_learning_pose_dataset", "models", model_subfolder))
    os.makedirs(local_output_dir, exist_ok=True)
    task.output_uri = local_output_dir

    # =========================================================================
    # 🐋 這裡的代碼只有當「背景 Agent 工人」咬到單之後，才會在背景全自動執行
    # =========================================================================
    print("====== 🍏 [Agent 遠端] 姿態辨識節點已成功接單，正式啟動重訓 ======")
    
    # 🎯 🌟 【業界滾動式重訓核心：繼承上一輪最強大腦】
    base_model_path = 'yolo11s-pose.pt'  # 預設起點：輕量級骨架辨識模型
    try:
        from clearml import Model
        current_proj = task.get_project_name()
        print(f"🔍 [增量鏈結] 正在檢查模型倉庫 '{current_proj}' 是否有上一輪產出的最強大腦...")
        
        # 這裡改為搜尋 'yolo' 與 'pose' 標籤，確保不會跟 RT-DETR 混淆！
        cloud_bests = Model.query_models(project_name=current_proj, tags=["yolo", "pose", "best"])
        if cloud_bests:
            cloud_bests = sorted(cloud_bests, key=lambda m: m.created, reverse=True)
            latest_cloud_model = cloud_bests[0]
            print(f"📥 [找到大腦] 發現上一輪的最新 YOLO-Pose 模型 (ID: {latest_cloud_model.id})，正在拉取權重進行繼承...")
            
            downloaded_base = latest_cloud_model.get_local_copy()
            if downloaded_base and os.path.exists(downloaded_base):
                base_model_path = downloaded_base
                print("🔄 [繼承成功] 成功載入最新模型權重，模型將在此基礎上『繼續進修』變更聰明！")
        else:
            print("ℹ️ 模型倉庫尚未有任何 'yolo' 'pose' 模型，本次重訓將從原始 'yolo11s-pose.pt' 開始冷啟動。")
    except Exception as e:
        print(f"⚠️ 嘗試拉取最新模型失敗 ({e})，降級使用原始 'yolo11s-pose.pt'。")
    
    # 2. 載入 YOLO 模型權重 (注意：這裡是 YOLO，不是 RTDETR！)
    model = YOLO(base_model_path)
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 未來的骨架資料集設定檔 (目前先預設為 pose_data.yaml)
    data_yaml_path = os.path.join(CURRENT_DIR, 'pose_data.yaml')

    # 3. 開始訓練
    import torch
    train_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 如果 pose_data.yaml 還沒準備好，這裡會報錯，不過這支腳本已經準備好了未來擴充的框架
    if os.path.exists(data_yaml_path):
        model.train(
            data=data_yaml_path, 
            epochs=10,             
            imgsz=640, 
            batch=8,               # 【Mac MPS 穩定設定】
            lr0=0.001,             
            patience=10,           
            device=train_device,   
            plots=False            
        )
    else:
        print(f"⚠️ 尚未建立骨架資料集的設定檔: {data_yaml_path}，請先準備好再啟動訓練！")
        return

    # =========================================================================
    # 🎯 動態綁定並強制為 ClearML 自動偵測到的模型貼上專屬標籤
    # =========================================================================
    models = task.get_models()
    
    if models and 'output' in models and len(models['output']) > 0:
        print("\n🚀 成功偵測到 ClearML 已自動捕捉到訓練產出的模型！")
        output_model = models['output'][-1]
        
        # 標籤改為 yolo 與 pose，防呆隔離！
        output_model.tags = ['yolo', 'pose', 'best']
        print("✅ [模型倉庫同步完成] 最新權重已成功上傳，並標記為 'yolo', 'pose', 'best' 標籤！\n")
    else:
        print("ℹ️ ClearML 未能自動捕捉模型，啟用 Fallback 機制尋找本地檔案...")
        local_best_path = "runs/pose/train/weights/best.pt"  # YOLO Pose 的預設輸出路徑
        
        if os.path.exists(local_best_path):
            output_model = OutputModel(
                task=task, 
                name="YOLO_Pose_Incremental_Training_Automated"
            )
            output_model.update_weights(weights_filename=local_best_path, auto_delete_local_copy=False)
            
            output_model.tags = ['yolo', 'pose', 'best']
            print("✅ [Fallback 同步完成] 已手動將本地模型推送至模型倉庫並標記 'yolo', 'pose', 'best'！\n")
        else:
            print("\n⚠️ 警告：找不到任何自動或本地的模型權重，請檢查訓練是否正常結束。\n")

if __name__ == "__main__":
    main()
