import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from clearml import Task, OutputModel
from ultralytics import RTDETR

def main():
    # 1. 初始化 Task
    # 當 Agent 背景執行時，這裡的 Task.init 會自動「接管」剛剛在 submit_task 建立好的排隊任務
    task = Task.init(
        project_name="Hazard_Detection", 
        task_name="RTDETR_Cloud_Incremental_Training_Automated"
    )
    
    # 🎯 🌟 [地端隔離分流] 依據專案類型將 YOLO-Pose 與 RT-DETR 模型分開儲存
    p_name = task.get_project_name() or "Fall_Detection"
    model_subfolder = "rt_detr" if "Hazard" in p_name or "DETR" in task.name else "yolo_pose"
    
    local_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "active_learning_dataset", "models", model_subfolder))
    os.makedirs(local_output_dir, exist_ok=True)
    task.output_uri = local_output_dir



    
    # =========================================================================
    # 🐋 這裡的代碼只有當「背景 Agent 工人」咬到單之後，才會在背景全自動執行
    # =========================================================================
    print("====== 🍏 [Agent 遠端] 背景運算節點已成功接單，正式啟動重訓 ======")
    
    # 🎯 🌟 【業界滾動式重訓核心：繼承上一輪最強大腦】
    # 不要直接寫死 rtdetr-l.pt！先去 ClearML 雲端尋找有沒有同專案、最新產出的 'best' 模型
    base_model_path = 'rtdetr-l.pt'
    old_map50 = 0.0

    try:
        from clearml import Model
        current_proj = task.get_project_name()
        print(f"🔍 [增量鏈結] 正在檢查模型倉庫 '{current_proj}' 是否有上一輪產出的最強大腦...")
        cloud_bests = Model.query_models(project_name=current_proj, tags=["detr", "best"])
        if cloud_bests:
            # 依據創建時間 (created) 在記憶體中排序，撈出最新產出的那個模型
            cloud_bests = sorted(cloud_bests, key=lambda m: m.created, reverse=True)
            latest_cloud_model = cloud_bests[0]
            print(f"📥 [找到大腦] 發現上一輪的最新 RT-DETR 模型 (ID: {latest_cloud_model.id})，正在拉取權重進行繼承...")
            
            # 讀取舊模型的 mAP50 標籤做為擂台基準
            for tag in latest_cloud_model.tags:
                if tag.startswith("map50_"):
                    try:
                        old_map50 = float(tag.replace("map50_", ""))
                    except ValueError:
                        pass
            print(f"🏆 [衛冕者成績] 舊模型的 mAP50 為 {old_map50:.4f}")
            
            # 下載該模型到 Agent 工作目錄下
            downloaded_base = latest_cloud_model.get_local_copy()
            if downloaded_base and os.path.exists(downloaded_base):
                base_model_path = downloaded_base
                print("🔄 [繼承成功] 成功載入最新模型權重，模型將在此基礎上『繼續進修』變更聰明！")
        else:
            print("ℹ️ 模型倉庫尚未有任何 'detr' 'best' 模型，本次重訓將從原始 'rtdetr-l.pt' 開始冷啟動。")
    except Exception as e:
        print(f"⚠️ 嘗試拉取最新模型失敗 ({e})，降級使用原始 'rtdetr-l.pt'。")
    
    # 2. 載入 RTDETR 模型權重（可能是原始的，也可能是上一輪傳下來的最強大腦）
    model = RTDETR(base_model_path)
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # =========================================================================
    # 🌟 動態資料切分 (Dynamic Data Splitting: 80% Train, 20% Valid)
    # 解決資料集未最佳化、可能導致過擬合 (Overfitting) 的問題
    # =========================================================================
    import shutil
    import random
    
    source_dir = os.path.join(CURRENT_DIR, "..", "active_learning_dataset")
    split_dir = os.path.join(CURRENT_DIR, "..", "active_learning_split_dataset")
    
    # 建立動態切分的目錄結構
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(split_dir, split, folder), exist_ok=True)
            
    # 讀取所有圖片
    source_images_dir = os.path.join(source_dir, "images")
    source_labels_dir = os.path.join(source_dir, "labels")
    
    if os.path.exists(source_images_dir):
        all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # =========================================================================
        # 🌟 平衡抽樣演算法 (Stratified Split by Rarest Class)
        # 解決類別不平衡問題，確保稀有類別在 Train/Val 中的比例一致
        # =========================================================================
        # 1. 讀取所有標籤，統計全域類別數量，並為每張照片找出出現的類別
        image_classes = {}
        global_class_counts = {}
        
        for img_name in all_images:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            src_txt = os.path.join(source_labels_dir, txt_name)
            classes_in_img = set()
            
            if os.path.exists(src_txt):
                with open(src_txt, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            classes_in_img.add(cls_id)
                            global_class_counts[cls_id] = global_class_counts.get(cls_id, 0) + 1
                            
            image_classes[img_name] = list(classes_in_img)
            
        # 2. 決定每張照片的「主類別 (Primary Class)」 (取該照片中包含的最稀有類別)
        # 若照片無標籤，歸類為 -1
        img_primary_class = {}
        for img, cls_list in image_classes.items():
            if not cls_list:
                img_primary_class[img] = -1
            else:
                # 依據 global_class_counts 排序，找出最稀有的類別
                rarest_class = min(cls_list, key=lambda c: global_class_counts.get(c, float('inf')))
                img_primary_class[img] = rarest_class
                
        # 3. 依據主類別進行分組
        grouped_images = {}
        for img, p_cls in img_primary_class.items():
            grouped_images.setdefault(p_cls, []).append(img)
            
        train_images = []
        val_images = []
        
        # 4. 對每個分組進行 80/20 切分 (保證平衡抽樣)
        for p_cls, imgs in grouped_images.items():
            random.seed(42 + p_cls) # 確保每次重訓的抽樣是可重現的
            random.shuffle(imgs)
            split_idx = int(len(imgs) * 0.8)
            train_images.extend(imgs[:split_idx])
            val_images.extend(imgs[split_idx:])
            
        print(f"📊 [平衡抽樣] 總共 {len(all_images)} 張相片。切分結果：Train={len(train_images)}, Valid={len(val_images)}")
        print(f"   -> 類別分佈狀態: {global_class_counts}")
        
        # 清空舊的暫存檔
        for folder in ['train', 'val']:
            for sub in ['images', 'labels']:
                d = os.path.join(split_dir, folder, sub)
                for f in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, f))
                    except Exception:
                        pass
                    
        # 複製檔案的內部函式
        def copy_files(file_list, split_name):
            for img_name in file_list:
                # 複製圖片
                src_img = os.path.join(source_images_dir, img_name)
                dst_img = os.path.join(split_dir, split_name, "images", img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                
                # 複製對應標籤
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                src_txt = os.path.join(source_labels_dir, txt_name)
                dst_txt = os.path.join(split_dir, split_name, "labels", txt_name)
                if os.path.exists(src_txt):
                    shutil.copy(src_txt, dst_txt)

        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
    else:
        print(f"⚠️ [警告] 找不到來源資料夾 {source_images_dir}，可能導致訓練失敗。")
    
    # 動態產生專用的 dynamic_data.yaml
    dynamic_yaml_path = os.path.join(CURRENT_DIR, 'dynamic_data.yaml')
    with open(dynamic_yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(split_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write("names:\n")
        f.write("  0: wheelchair\n")
        f.write("  1: bed\n")
        
    print(f"✅ [動態切分] dynamic_data.yaml 產生完成！路徑：{dynamic_yaml_path}")

    # 3. 開始訓練
    # 🎯 直接在 train 內加入 plots=False，這能 100% 關閉大圖生成與上傳，
    # 同時完美避開了 ImportError 版本相容問題，並大幅節省連線頻寬！
    import torch
    train_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.train(
        data=dynamic_yaml_path, 
        epochs=10,             # 【實戰設定】設為 10 輪 (快速高效重訓)
        imgsz=640, 
        batch=8,               # 【Mac MPS 穩定設定】批次大小 8 避免顯存溢出
        lr0=0.001,             # 【實戰設定】微調學習率
        patience=10,           # 【實戰設定】早停機制
        device=train_device,   # 自動偵測 GPU/MPS 加速
        plots=False,           # 阻擋生成/上傳大圖，確保連線不中斷
        project="runs/detect",
        name="train",
        exist_ok=True
    )

    # 4. 讀取訓練後的成績 (Challenger mAP50)
    import pandas as pd
    new_map50 = 0.0
    csv_path = "runs/detect/train/results.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if "metrics/mAP50(B)" in df.columns:
                new_map50 = df.iloc[-1]["metrics/mAP50(B)"]
                print(f"📊 [挑戰者成績] 剛訓練完的新模型 mAP50 為 {new_map50:.4f}")
        except Exception as e:
            print(f"⚠️ 解析 results.csv 失敗: {e}")

    # =========================================================================
    # 5. 打擂台機制 (Champion vs Challenger) 與 Discord Webhook
    # =========================================================================
    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1532199171531083846/OUJSFZjAw710l7Szw66hrzspo3aGfniGves8pP1nUSQ1x9EOAzP9yJT1wmATMg7yx_Bt")

    def send_discord_notification(title, message, color):
        if not WEBHOOK_URL:
            return
        try:
            import requests
            data = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": color
                }]
            }
            requests.post(WEBHOOK_URL, json=data, timeout=5)
        except Exception as e:
            print(f"⚠️ 發送 Discord 通知失敗: {e}")

    models = task.get_models()
    output_model = None
    
    if models and 'output' in models and len(models['output']) > 0:
        print("\n🚀 成功偵測到 ClearML 已自動捕捉到訓練產出的模型！")
        output_model = models['output'][-1]
    else:
        # 💡 Fallback 安全備份機制
        print("ℹ️ ClearML 未能自動捕捉模型，啟用 Fallback 機制尋找本地檔案...")
        local_best_path = "runs/detect/train/weights/best.pt"
        if os.path.exists(local_best_path):
            output_model = OutputModel(task=task, name="RTDETR_Cloud_Incremental_Training_Automated")
            output_model.update_weights(weights_filename=local_best_path, auto_delete_local_copy=False)

    if output_model:
        # 動態貼標與打擂台
        map_tag = f"map50_{new_map50:.4f}"
        
        if new_map50 >= old_map50:
            print(f"🎉 打擂台成功！新模型 ({new_map50:.4f}) 擊敗或平手舊模型 ({old_map50:.4f})")
            output_model.tags = ['detr', 'best', map_tag]
            
            # 發送成功通知 (綠色: 5763719)
            msg = f"**新模型 mAP50**: `{new_map50:.4f}` 🏆 (超越或持平舊版 `{old_map50:.4f}`)\n**狀態**: 已自動標記為 `best`，Edge 端即將自動更新！\n**Task ID**: `{task.id}`"
            send_discord_notification("🎉 【模型重訓成功：自動部署過關】", msg, 5763719)
        else:
            print(f"⚠️ 打擂台失敗！新模型 ({new_map50:.4f}) 遜於舊模型 ({old_map50:.4f})")
            output_model.tags = ['detr', 'Draft', map_tag]
            
            # 發送失敗通知 (紅色: 15548997)
            msg = f"**新模型 mAP50**: `{new_map50:.4f}` ❌ (低於舊版 `{old_map50:.4f}`)\n**狀態**: 模型表現退步，已自動廢棄並維持原冠軍模型運作。\n**Task ID**: `{task.id}`"
            send_discord_notification("⚠️ 【模型重訓警告：自動阻擋部署】", msg, 15548997)

        print(f"✅ [模型倉庫同步完成] 權重處理完畢，目前標籤: {output_model.tags}\n")
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