import os
import sys
from pathlib import Path
import torch
import json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from modules.action_transformer import ActionTransformer

MODEL_DIR = PROJECT_ROOT / "models" / "action_classifier"
PT_MODEL_PATH = MODEL_DIR / "transformer_action_model.pt"
ONNX_MODEL_PATH = MODEL_DIR / "transformer_action_model.onnx"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

# 推理參數
SEQ_LENGTH = 30
INPUT_DIM = 34

def main():
    if not PT_MODEL_PATH.exists():
        print(f"❌ 找不到訓練好的 PyTorch 模型: {PT_MODEL_PATH}")
        sys.exit(1)
        
    if not LABEL_MAP_PATH.exists():
        print(f"❌ 找不到標籤對照表: {LABEL_MAP_PATH}")
        sys.exit(1)
        
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    print("🔄 正在實例化 Action Transformer 模型...")
    # 建立模型結構
    model = ActionTransformer(num_classes=num_classes, input_dim=INPUT_DIM, d_model=64, nhead=4, num_layers=2)
    
    # 載入權重
    print(f"📥 載入權重: {PT_MODEL_PATH}")
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval() # 設定為推理模式 (重要：關閉 Dropout 等)
    
    # 建立 Dummy Input (模擬推論時輸入的 Tensor)
    # 形狀: (Batch, Sequence_Length, Features)
    # 我們設定 dynamic_axes 讓 Batch 大小在 Triton 部署時可以動態改變
    dummy_input = torch.randn(1, SEQ_LENGTH, INPUT_DIM, requires_grad=False)
    
    print("🚀 開始匯出 ONNX 格式...")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=14, # 確保相容較新的 ONNX 算子
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['action_logits'],
        dynamic_axes={
            'input_sequence': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        }
    )
    
    print(f"\n🎉 匯出成功！ONNX 模型已儲存至: {ONNX_MODEL_PATH}")
    
    # 順便產生一份基礎的 Triton config.pbtxt 給用戶
    config_pbtxt = f"""
name: "action_transformer"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {{
    name: "input_sequence"
    data_type: TYPE_FP32
    dims: [ {SEQ_LENGTH}, {INPUT_DIM} ]
  }}
]

output [
  {{
    name: "action_logits"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
  }}
]
"""
    config_path = MODEL_DIR / "config.pbtxt"
    with open(config_path, "w") as f:
        f.write(config_pbtxt.strip())
        
    print(f"📄 已為您產生 Triton 模型設定檔草稿: {config_path}")

if __name__ == "__main__":
    main()
