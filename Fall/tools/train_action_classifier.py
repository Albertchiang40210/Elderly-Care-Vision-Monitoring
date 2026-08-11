import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from mlops_config.settings import settings

# 嘗試載入 ClearML (如果您有設定好環境變數)
try:
    from clearml import Task
    clearml_available = True
except ImportError:
    clearml_available = False
    print("⚠️ 找不到 ClearML 套件，將進行本地訓練。")

from modules.action_transformer import ActionTransformer

DATASET_PATH = PROJECT_ROOT / "active_learning_dataset" / "action_features" / "action_dataset.npz"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "action_classifier"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "transformer_action_model.pt"
CHALLENGER_MODEL_PATH = MODEL_SAVE_DIR / "transformer_action_model_challenger.pt"
LABEL_MAP_PATH = MODEL_SAVE_DIR / "label_map.json"
METRICS_PATH = MODEL_SAVE_DIR / "metrics.json"

# 超參數設定由 settings 集中管理
BATCH_SIZE = settings.BATCH_SIZE
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE
SEQ_LENGTH = settings.SEQ_LENGTH
INPUT_DIM = settings.INPUT_DIM

def main():
    if not DATASET_PATH.exists():
        print(f"❌ 找不到特徵資料集: {DATASET_PATH}")
        print("💡 請先等待提取特徵完畢！")
        sys.exit(1)

    print("🔄 載入動作特徵資料集...")
    data = np.load(DATASET_PATH)
    X = data['X'] # 原本是 flatten 的 [N, 1020]
    y = data['y']
    
    # 將壓平的特徵重新 Reshape 成 Transformer 需要的格式 (Batch, Sequence, Feature)
    X = X.reshape(-1, SEQ_LENGTH, INPUT_DIM)
    
    print(f"📊 資料集大小: {X.shape[0]} 筆特徵序列, 形狀: {X.shape}")
    
    # 類別轉換
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    inverse_label_map = {idx: label for label, idx in label_map.items()}
    y_encoded = np.array([label_map[label] for label in y])
    
    # 儲存 Label Map 供推論時使用
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(inverse_label_map, f)
    
    print(f"🏷️ 類別映射: {label_map}")

    # 切割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 轉換為 PyTorch Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用運算裝置: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化 Action Transformer
    model = ActionTransformer(
        num_classes=len(unique_labels),
        input_dim=INPUT_DIM,
        d_model=settings.D_MODEL,
        nhead=settings.NHEAD,
        num_layers=settings.NUM_LAYERS
    ).to(device)
    
    # ♻️ 新增：嘗試載入歷史權重進行增量學習 (Incremental Learning)
    if MODEL_SAVE_PATH.exists():
        print(f"♻️ 發現既有模型權重 ({MODEL_SAVE_PATH.name})，準備進行增量訓練 (Fine-tuning)...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print("✅ 成功載入歷史權重！不須從頭訓練。")
        except Exception as e:
            print(f"⚠️ 載入歷史權重失敗 (可能是有新增全新的動作類別導致架構改變): {e}")
            print("➡️ 將從頭開始訓練 (Train from scratch)。")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 啟動 ClearML 紀錄
    task = None
    if clearml_available:
        try:
            task = Task.init(project_name="Fall_Detection_Action", task_name="Train_Transformer_Classifier", reuse_last_task_id=False)
            task.connect({
                "epochs": EPOCHS, 
                "batch_size": BATCH_SIZE, 
                "lr": LEARNING_RATE, 
                "d_model": settings.D_MODEL,
                "nhead": settings.NHEAD,
                "num_layers": settings.NUM_LAYERS
            })
        except Exception as e:
            print(f"⚠️ ClearML 連線失敗，略過紀錄: {e}")

    print("🔥 開始訓練 Action Transformer ...")
    
    historical_best_acc = 0.0
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            historical_best_acc = json.load(f).get("best_acc", 0.0)
    print(f"🛡️ 目前衛冕者 (Champion) 準確率: {historical_best_acc * 100:.2f}%")
    
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 驗證
        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {acc*100:.2f}%")
        
        if task:
            task.get_logger().report_scalar("Loss", "Train", value=train_loss/len(train_loader), iteration=epoch)
            task.get_logger().report_scalar("Loss", "Val", value=val_loss/len(test_loader), iteration=epoch)
            task.get_logger().report_scalar("Metrics", "Accuracy", value=acc, iteration=epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CHALLENGER_MODEL_PATH)

    print(f"\n🏆 本次訓練最佳驗證集準確率 (Challenger Best Accuracy): {best_acc * 100:.2f}%")
    
    # 擂台賽對決 (Champion vs Challenger)
    if best_acc > historical_best_acc:
        print(f"🎉 挑戰成功！新模型準確率 ({best_acc*100:.2f}%) 超越舊模型 ({historical_best_acc*100:.2f}%)")
        print("✅ 正在部署新模型為 Champion...")
        os.replace(CHALLENGER_MODEL_PATH, MODEL_SAVE_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump({"best_acc": best_acc}, f)
    else:
        print(f"❌ 挑戰失敗！新模型準確率 ({best_acc*100:.2f}%) 未能超越舊模型 ({historical_best_acc*100:.2f}%)")
        print("🛡️ 保留原有 Champion 模型，捨棄本次訓練結果。")
        if CHALLENGER_MODEL_PATH.exists():
            os.remove(CHALLENGER_MODEL_PATH)
        sys.exit(1) # 挑戰失敗，拋出例外以中止後續部署流程

if __name__ == "__main__":
    main()
