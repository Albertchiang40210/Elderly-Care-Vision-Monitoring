import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*50)
    print("🚀 啟動 ClearML 動作分類器 (Action Transformer) 全自動重訓流程...")
    print("="*50)

    # 1. 特徵提取 (YOLO-Pose)
    print("\n" + "="*40)
    print("1️⃣ 進行骨架特徵提取 (YOLO-Pose)...")
    print("="*40)
    script_1 = os.path.join(base_dir, "extract_action_features.py")
    subprocess.check_call([sys.executable, script_1])

    # 2. 擂台賽對決訓練
    print("\n" + "="*40)
    print("2️⃣ 訓練 Action Transformer (擂台賽對決)...")
    print("="*40)
    script_2 = os.path.join(base_dir, "train_action_classifier.py")
    subprocess.check_call([sys.executable, script_2])

    # 3. 匯出模型
    print("\n" + "="*40)
    print("3️⃣ 匯出 ONNX 並準備熱更新...")
    print("="*40)
    script_3 = os.path.join(base_dir, "export_action_model.py")
    subprocess.check_call([sys.executable, script_3])

    # 4. 熱更新到 Triton
    print("\n" + "="*40)
    print("4️⃣ 發動 MLOps 代理人，進行 Triton 熱部署...")
    print("="*40)
    script_4 = os.path.join(base_dir, "model_deployment_agent.py")
    subprocess.check_call([sys.executable, script_4, "action_classifier"])

    print("\n" + "="*50)
    print("🎉 Action Transformer 全自動重訓與熱更新成功完成！")
    print("="*50)

if __name__ == "__main__":
    main()
