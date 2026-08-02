import os
import sys
import subprocess
import json
import requests

def send_discord_notification(title, message, color):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1532199171531083846/OUJSFZjAw710l7Szw66hrzspo3aGfniGves8pP1nUSQ1x9EOAzP9yJT1wmATMg7yx_Bt")
    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color
            }]
        }
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"⚠️ 發送 Discord 通知失敗: {e}")

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
    try:
        subprocess.check_call([sys.executable, script_2])
    except subprocess.CalledProcessError:
        print("\n⚠️ 擂台賽挑戰失敗，中止後續匯出與部署流程。")
        send_discord_notification("⚠️ 【動作分類器重訓警告：自動阻擋部署】", "新訓練的 Action Transformer 模型未能超越目前衛冕者，已自動放棄部署，維持原系統穩定運作。", 15548997)
        sys.exit(0)

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

    # 5. 發送勝利 Discord 通知
    metrics_file = os.path.join(base_dir, "..", "models", "action_classifier", "metrics.json")
    acc = "未知"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            acc = f"{json.load(f).get('best_acc', 0) * 100:.2f}%"
    
    msg = f"**動作分類器 (Action Transformer) 重訓完畢！**\n✅ 已自動完成資料萃取、擂台賽對決，並熱更新至 Triton。\n🎯 最新 Champion 模型準確率: `{acc}`"
    send_discord_notification("🎉 【Action Transformer 重訓成功：自動部署過關】", msg, 5763719)
    print("✅ 成功發送 Discord 部署捷報！")

if __name__ == "__main__":
    main()
