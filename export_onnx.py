from ultralytics import RTDETR
import os
import shutil

print("Step 1: 正在載入 rtdetr-l.pt 模型...")
model = RTDETR("Fall/rtdetr-l.pt")

print("Step 2: 正在將模型轉成 ONNX 格式 (強制指定 opset=16)...")
model.export(format="onnx", imgsz=[640, 640], dynamic=False, opset=16)

# 自動幫你把轉好的檔案移到正確的位置
src_path = "rtdetr-l.onnx"
dest_path = "Fall/model_repository/rt_detr/1/model.onnx"

print("Step 3: 整理檔案中...")
if os.path.exists(src_path):
    if os.path.exists(dest_path):
        os.remove(dest_path)
    shutil.move(src_path, dest_path)
    print("🎉 大成功！相容版 ONNX 檔案已經自動放到：Fall/model_repository/rt_detr/1/model.onnx")
else:
    print("❌ 轉檔失敗，請確認你的 Fall 目錄下是否有 rtdetr-l.pt 檔案！")

# =========================================================================
# 💡 [檔案說明與核心職責]
# 「它是模型跨平台轉檔與 Triton 倉庫部署工具 (ONNX Model Exporter)。」
# 本腳本用於將訓練好的 PyTorch RT-DETR 模型 (.pt) 匯出轉檔為 ONNX 格式：
# 1. 載入 Fall/rtdetr-l.pt 模型權重。
# 2. 強制指定 opset=16 導出為 ONNX 跨框架相容格式。
# 3. 自動將生成的 model.onnx 移至 Triton Inference Server 的 model_repository 中，解鎖硬體加速推論。
# =========================================================================

