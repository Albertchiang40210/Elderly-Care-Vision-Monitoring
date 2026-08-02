import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from clearml import Task, OutputModel
from ultralytics import YOLO

def main():
    # 0. 訓練前，自動確保從 Label Studio 倒出最新的人工標註資料 (防呆機制)
    print("🔄 [自動防呆] 正在從 Label Studio 匯出最新標註資料...")
    os.system("python Fall/tools/export_pose_annotations.py")
    
    # 1. 初始化 Task
    # 這是專為「人體姿態/跌倒辨識」打造的專屬高速公路！
    task = Task.init(
        project_name="Fall_Detection", 
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
    old_map50 = 0.0

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
            
            # 讀取舊模型的 mAP50 標籤做為擂台基準
            for tag in latest_cloud_model.tags:
                if tag.startswith("map50_"):
                    try:
                        old_map50 = float(tag.replace("map50_", ""))
                    except ValueError:
                        pass
            print(f"🏆 [衛冕者成績] 舊模型的 mAP50 為 {old_map50:.4f}")
            
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
    
    # =========================================================================
    # 🌟 平衡抽樣演算法 (Stratified Split by Rarest Class)
    # 解決類別不平衡問題，確保稀有類別在 Train/Val 中的比例一致
    # =========================================================================
    import shutil
    import random
    
    source_dir = os.path.join(CURRENT_DIR, "..", "active_learning_pose_dataset")
    split_dir = os.path.join(CURRENT_DIR, "..", "active_learning_pose_split_dataset")
    
    # 建立動態切分的目錄結構
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(split_dir, split, folder), exist_ok=True)
            
    # 讀取所有圖片
    source_images_dir = os.path.join(source_dir, "images")
    source_labels_dir = os.path.join(source_dir, "labels")
    
    if os.path.exists(source_images_dir):
        all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
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
            
        # 2. 決定每張照片的「主類別 (Primary Class)」
        img_primary_class = {}
        for img, cls_list in image_classes.items():
            if not cls_list:
                img_primary_class[img] = -1
            else:
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
            random.seed(42 + p_cls)
            random.shuffle(imgs)
            split_idx = int(len(imgs) * 0.8)
            train_images.extend(imgs[:split_idx])
            val_images.extend(imgs[split_idx:])
            
        print(f"📊 [Pose 平衡抽樣] 總共 {len(all_images)} 張相片。切分結果：Train={len(train_images)}, Valid={len(val_images)}")
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
                    
        # 複製檔案
        def copy_files(file_list, split_name):
            for img_name in file_list:
                src_img = os.path.join(source_images_dir, img_name)
                dst_img = os.path.join(split_dir, split_name, "images", img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                src_txt = os.path.join(source_labels_dir, txt_name)
                dst_txt = os.path.join(split_dir, split_name, "labels", txt_name)
                if os.path.exists(src_txt):
                    shutil.copy(src_txt, dst_txt)

        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
    else:
        print(f"⚠️ [警告] 找不到來源資料夾 {source_images_dir}，可能導致訓練失敗。")

    # 動態產生 YOLO-Pose 專用的 dynamic_data.yaml
    dynamic_yaml_path = os.path.join(CURRENT_DIR, 'pose_dynamic_data.yaml')
    with open(dynamic_yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(split_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write("nc: 1\n")
        f.write("names: ['person']\n")
        f.write("kpt_shape: [17, 3]\n")
        
    print(f"✅ [動態切分] pose_dynamic_data.yaml 產生完成！路徑：{dynamic_yaml_path}")

    # 3. 開始訓練
    import torch
    train_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.train(
        data=dynamic_yaml_path, 
        epochs=30,             
        imgsz=640, 
        batch=8,               # 【Mac MPS 穩定設定】
        lr0=0.001,             
        patience=10,           
        device=train_device,   
        plots=False,
        project="runs/pose",
        name="train",
        exist_ok=True
    )

    # 4. 讀取訓練後的成績 (Challenger mAP50)
    import pandas as pd
    new_map50 = 0.0
    csv_path = f"{model.trainer.save_dir}/results.csv" if getattr(model, 'trainer', None) else "runs/pose/train/results.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            # Pose 模型通常會有 Pose mAP50(P) 或是 Box mAP50(B)
            if "metrics/mAP50(P)" in df.columns:
                new_map50 = df.iloc[-1]["metrics/mAP50(P)"]
            elif "metrics/mAP50(B)" in df.columns:
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
        print("ℹ️ ClearML 未能自動捕捉模型，啟用 Fallback 機制尋找本地檔案...")
        local_best_path = "runs/pose/train/weights/best.pt"
        if os.path.exists(local_best_path):
            output_model = OutputModel(task=task, name="YOLO_Pose_Incremental_Training_Automated")
            output_model.update_weights(weights_filename=local_best_path, auto_delete_local_copy=False)

    if output_model:
        # 動態貼標與打擂台
        map_tag = f"map50_{new_map50:.4f}"
        
        if new_map50 >= old_map50:
            print(f"🎉 打擂台成功！新模型 ({new_map50:.4f}) 擊敗或平手舊模型 ({old_map50:.4f})")
            output_model.tags = ['yolo', 'pose', 'best', map_tag]
            
            msg = f"**新模型 mAP50**: `{new_map50:.4f}` 🏆 (超越或持平舊版 `{old_map50:.4f}`)\n**狀態**: 已自動標記為 `best`，Edge 端即將自動更新！\n**Task ID**: `{task.id}`"
            send_discord_notification("🎉 【YOLO-Pose 骨架重訓成功：自動部署過關】", msg, 5763719)
        else:
            print(f"⚠️ 打擂台失敗！新模型 ({new_map50:.4f}) 遜於舊模型 ({old_map50:.4f})")
            output_model.tags = ['yolo', 'pose', 'Draft', map_tag]
            
            msg = f"**新模型 mAP50**: `{new_map50:.4f}` ❌ (低於舊版 `{old_map50:.4f}`)\n**狀態**: 模型表現退步，已自動廢棄並維持原冠軍模型運作。\n**Task ID**: `{task.id}`"
            send_discord_notification("⚠️ 【YOLO-Pose 骨架重訓警告：自動阻擋部署】", msg, 15548997)

        print(f"✅ [模型倉庫同步完成] 權重處理完畢，目前標籤: {output_model.tags}\n")
    else:
        print("\n⚠️ 警告：找不到任何自動或本地的模型權重，請檢查訓練是否正常結束。\n")

if __name__ == "__main__":
    main()
