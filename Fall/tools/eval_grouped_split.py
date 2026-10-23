"""以「連續片段」為單位切分訓練/驗證集，檢驗隨機切分是否高估準確率。

action_dataset.npz 只存 X、y，沒有影片編號；但特徵是逐支影片依序提取，
因此同一標籤的連續片段（run）可近似視為同一支影片。整個 run 只會出現在
訓練或驗證其中一邊，避免滑動窗口重疊造成的資料洩漏。
用法: python eval_grouped_split.py [epochs]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlops_config.settings import settings
from modules.action_transformer import ActionTransformer

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
torch.manual_seed(0)
np.random.seed(0)

d = np.load(ROOT / "active_learning_dataset/action_features/action_dataset.npz", allow_pickle=True)
X = d["X"].reshape(-1, settings.SEQ_LENGTH, settings.INPUT_DIM).astype(np.float32)
labels = np.unique(d["y"])
y = np.array([list(labels).index(v) for v in d["y"]])

# 連續相同標籤視為同一組
group = np.concatenate([[0], np.cumsum(y[1:] != y[:-1])])
g_ids = np.unique(group)
g_label = np.array([y[group == g][0] for g in g_ids])
tr_g, te_g = train_test_split(g_ids, test_size=0.2, random_state=42, stratify=g_label)
tr_m, te_m = np.isin(group, tr_g), np.isin(group, te_g)
print(f"片段數 {len(g_ids)}｜訓練 {tr_m.sum()} 筆（{len(tr_g)} 段）｜驗證 {te_m.sum()} 筆（{len(te_g)} 段）")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionTransformer(len(labels), settings.INPUT_DIM, settings.D_MODEL, settings.NHEAD, settings.NUM_LAYERS).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
tl = DataLoader(TensorDataset(torch.tensor(X[tr_m]), torch.tensor(y[tr_m])), batch_size=settings.BATCH_SIZE, shuffle=True)
Xte, yte = torch.tensor(X[te_m]).to(dev), y[te_m]

def evaluate():
    model.eval()
    with torch.no_grad():
        return model(Xte).argmax(1).cpu().numpy()

for ep in range(1, EPOCHS + 1):
    model.train()
    for bx, by in tl:
        opt.zero_grad()
        loss_fn(model(bx.to(dev)), by.to(dev)).backward()
        opt.step()
    if ep % 5 == 0 or ep == EPOCHS:
        print(f"epoch {ep:3d}  val acc {accuracy_score(yte, evaluate()) * 100:.2f}%", flush=True)

print("\n最後一個 epoch 的分類報告（非挑最佳 epoch）：")
print(classification_report(yte, evaluate(), target_names=list(labels), digits=3))
