import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from mlops_config.settings import settings
from modules.action_transformer import ActionTransformer

DATASET_PATH = PROJECT_ROOT / "active_learning_dataset" / "action_features" / "action_dataset.npz"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "action_classifier"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "transformer_action_model.pt"
LABEL_MAP_PATH = MODEL_SAVE_DIR / "label_map.json"
OUTPUT_DIR = PROJECT_ROOT / "presentation_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 解決 matplotlib 中文字型顯示問題
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC', 'sans-serif'] # 支援多種 Mac 中文字體
plt.rcParams['axes.unicode_minus'] = False

def generate_confusion_matrix():
    print("🔄 正在生成 Action Transformer 混淆矩陣...")
    if not DATASET_PATH.exists() or not MODEL_SAVE_PATH.exists():
        print("❌ 找不到特徵集或模型，無法產生混淆矩陣。")
        return

    # 1. 載入資料
    data = np.load(DATASET_PATH)
    X = data['X'].reshape(-1, settings.SEQ_LENGTH, settings.INPUT_DIM)
    y = data['y']

    with open(LABEL_MAP_PATH, "r") as f:
        inverse_label_map = json.load(f)
    label_map = {v: int(k) for k, v in inverse_label_map.items()}
    y_encoded = np.array([label_map[label] for label in y])

    # 2. 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionTransformer(
        num_classes=len(label_map),
        input_dim=settings.INPUT_DIM,
        d_model=settings.D_MODEL,
        nhead=settings.NHEAD,
        num_layers=settings.NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 3. 預測
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y_encoded, preds)
    print(f"✅ 模型整體準確率: {acc*100:.2f}%")

    # 4. 畫圖
    cm = confusion_matrix(y_encoded, preds)
    labels = [inverse_label_map[str(i)] for i in range(len(label_map))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Action Transformer Confusion Matrix (Acc: {acc*100:.1f}%)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    out_path = OUTPUT_DIR / "action_confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 混淆矩陣已儲存至: {out_path}")

if __name__ == "__main__":
    generate_confusion_matrix()
