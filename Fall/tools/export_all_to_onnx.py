import os
import torch
import torch.nn as nn
from ultralytics import YOLO

# 定義 ActionTransformer 結構與 inference_test.py 保持完全一致
class ActionTransformer(nn.Module):
    def __init__(self, input_dim=34, seq_len=30, num_classes=2):
        super(ActionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Fall/
    
    # 建立 model_repository 目錄
    yolo_repo_dir = os.path.join(base_dir, "model_repository", "yolo_pose", "1")
    act_repo_dir = os.path.join(base_dir, "model_repository", "action_transformer", "1")
    os.makedirs(yolo_repo_dir, exist_ok=True)
    os.makedirs(act_repo_dir, exist_ok=True)
    
    print("--------------------------------------------------")
    print("Step 1: 正在匯出 YOLO Pose (yolo11s-pose.pt) 至 ONNX...")
    # YOLO Pose 檔案路徑
    yolo_pt_path = os.path.join(base_dir, "yolo11s-pose.pt")
    if os.path.exists(yolo_pt_path):
        model_yolo = YOLO(yolo_pt_path)
        # 導出為 ONNX 格式，固定輸入尺寸 640x640，opset=16 確保相容性
        yolo_onnx_path = model_yolo.export(format="onnx", imgsz=[640, 640], opset=16, dynamic=False)
        dest_yolo_onnx = os.path.join(yolo_repo_dir, "model.onnx")
        if os.path.exists(yolo_onnx_path):
            if os.path.exists(dest_yolo_onnx):
                os.remove(dest_yolo_onnx)
            os.rename(yolo_onnx_path, dest_yolo_onnx)
            print(f"✅ YOLO Pose ONNX 成功儲存至: {dest_yolo_onnx}")
        else:
            print("❌ 找不到導出的 YOLO Pose ONNX 檔案")
    else:
        print(f"❌ 找不到 YOLO Pose pt 檔: {yolo_pt_path}")

    print("--------------------------------------------------")
    print("Step 2: 正在匯出 ActionTransformer 至 ONNX...")
    act_pth_path = os.path.join(base_dir, "tools", "action_transformer.pth")
    if os.path.exists(act_pth_path):
        model_act = ActionTransformer()
        model_act.load_state_dict(torch.load(act_pth_path, map_location="cpu"))
        model_act.eval()
        
        # 虛擬輸入 [batch_size, seq_len=30, input_dim=34]
        dummy_input = torch.randn(1, 30, 34, dtype=torch.float32)
        dest_act_onnx = os.path.join(act_repo_dir, "model.onnx")
        
        torch.onnx.export(
            model_act,
            dummy_input,
            dest_act_onnx,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=16
        )
        
        # 修正 Triton 23.10 不支援 IR Version 10 的問題，將其降級為 8
        import onnx
        onnx_model = onnx.load(dest_act_onnx)
        onnx_model.ir_version = 8
        onnx.save(onnx_model, dest_act_onnx)
        print(f"🔄 已成功將 ActionTransformer ONNX IR 版本降級至 8 以相容於 Triton 23.10")
        print(f"✅ ActionTransformer ONNX 成功儲存至: {dest_act_onnx}")
    else:
        print(f"❌ 找不到 ActionTransformer pth 檔: {act_pth_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
