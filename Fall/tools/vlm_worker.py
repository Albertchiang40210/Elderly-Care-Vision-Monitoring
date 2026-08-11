import json
import time
import os
import re
import shutil
import ollama  
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import RTDETR  
import boto3
import cv2
import numpy as np
from typing import TypedDict, Optional, List, Dict, Any

# 引入 LangGraph 核心元件
from langgraph.graph import StateGraph, START, END

# =========================================================================
# 💡 核心大腦雙引擎宣告對齊 (加入智慧動態加載 MLOps 最新模型機制)
# =========================================================================
VLM_MODEL_NAME = "qwen2.5vl:latest"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # 從 tools/ 往上推一層到 Fall/

# 🌟 定義部署代理人（Agent）會自動下載並覆蓋更新的「固定黃金模型路徑」
ACTIVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model_repository", "rt_detr", "2", "model.onnx")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "rtdetr-l.pt")

# 智慧判斷：如果 Agent 已經下載了重訓後的新模型，就自動加載它！否則用預設基礎模型
if os.path.exists(ACTIVE_MODEL_PATH):
    print(f"🔥 [VLM 偵測] 發現 MLOps 部署代理人已下載新重訓大腦！正在載入：{ACTIVE_MODEL_PATH}")
    DETECTOR_MODEL = RTDETR(ACTIVE_MODEL_PATH)
else:
    print(f"ℹ️  [VLM 偵測] 未偵測到新重訓模型，採用預設基礎模型：{DEFAULT_MODEL_PATH}")
    DETECTOR_MODEL = RTDETR(DEFAULT_MODEL_PATH)

# =========================================================================
# 📝 1. 定義 LangGraph 狀態 (AgentState)
# =========================================================================
class AgentState(TypedDict):
    event_data: Dict[str, Any]
    alert_type: str
    env_clues: str
    cam_id: str
    clean_device_id: int
    clip_path: str
    img_path: str
    local_backup_img: str
    is_s3: bool
    highest_score: float
    best_box: Optional[Any]
    vlm_input_source: List[str]
    raw_report: Optional[str]
    severity: str
    should_send_report: bool
    vlm_fall_reason_item: str
    vlm_item_description: str

# =========================================================================
# 📹 2. 輔助函數：影格抽取
# =========================================================================
def extract_key_frames(video_path, num_frames=5):
    """從影片中均勻抽取指定數量的影格，並縮小解析度以大幅加速 VLM 推理"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    temp_frames = []
    temp_dir = os.path.join(PROJECT_ROOT, "temp_vlm_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, (480, 360))
            frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, resized)
            temp_frames.append(frame_path)
            
    cap.release()
    return temp_frames

# =========================================================================
# 🧠 3. 輔助函數：主動學習打包引擎
# =========================================================================
def package_active_learning_sample(img_path, camera_id, rtdetr_box_data, vlm_inferred_label=None, local_backup_path=None, category="false_alarms"):
    """Label Studio 原生預測 JSON 打包與 S3 分類資料夾歸聯"""
    try:
        if category == "false_alarms":
            # 這是 YOLO-Pose 的跌倒誤報
            dataset_base = os.path.join(PROJECT_ROOT, "label_studio_data", "pose_false_alarms")
        elif category == "hazard_objects":
            # 這是 DETR 的輪椅與床鋪新資料
            dataset_base = os.path.join(PROJECT_ROOT, "label_studio_data", "detr_hazard_objects")
        else:
            dataset_base = os.path.join(PROJECT_ROOT, "label_studio_data", category)
            
        os.makedirs(os.path.join(dataset_base, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_base, "predictions"), exist_ok=True)
        
        base_name = os.path.basename(img_path)
        json_name = base_name + ".json"
        
        target_img_path = f"{dataset_base}/images/{base_name}"
        real_local_src = None
        if img_path.startswith("s3://"):
            if local_backup_path and os.path.exists(local_backup_path):
                real_local_src = local_backup_path
                if os.path.abspath(local_backup_path) != os.path.abspath(target_img_path):
                    shutil.copy(local_backup_path, target_img_path)
            else:
                print(f"⚠️ [方案 B] 檢測到 S3 網址且無本地備份，將跳過實體圖片複製")
        else:
            real_local_src = img_path
            if os.path.abspath(img_path) != os.path.abspath(target_img_path):
                shutil.copy(img_path, target_img_path)

        # 🚀 目前無 AWS 環境，直接跳過 S3 上傳，僅保留在本地
        # if real_local_src and os.path.exists(real_local_src):
        #     try:
        #         s3_c = boto3.client('s3')
        #         bucket_n = os.getenv("AWS_BUCKET_NAME", "aipe03-3")
        #         s3_key = f"active_learning/{category}/{base_name}"
        #         s3_c.upload_file(real_local_src, bucket_n, s3_key)
        #         print(f"☁️ [S3 自動歸類] 成功將照片上傳至 S3 專屬資料夾: s3://{bucket_n}/{s3_key}")
        #     except Exception as s3_err:
        #         print(f"⚠️ 上傳 S3 專屬資料夾失敗 (仍保存在本地): {s3_err}")

        
        if not rtdetr_box_data:
            return

        cls_id = int(rtdetr_box_data.cls[0])
        xywhn = rtdetr_box_data.xywhn[0].tolist()
        
        x_max_100 = (xywhn[0] - (xywhn[2] / 2)) * 100
        y_max_100 = (xywhn[1] - (xywhn[3] / 2)) * 100
        w_max_100 = xywhn[2] * 100
        h_max_100 = xywhn[3] * 100
        
        label_map = {0: "wheelchair", 1: "slipper", 2: "wire", 3: "obstacle", 4: "walker"}
        ALLOWED_HAZARDS = {"wheelchair", "slipper", "wire", "obstacle", "walker"}

        
        if vlm_inferred_label and vlm_inferred_label.lower() in ALLOWED_HAZARDS:
            label_name = vlm_inferred_label.lower()
            print(f"🏷️  [主動學習] VLM 自動為偵測框套用通過白名單之危險物品標籤: '{label_name}'")
        else:
            label_name = label_map.get(cls_id, "obstacle")


        label_studio_json = {
            "result": [
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_max_100,
                        "y": y_max_100,
                        "width": w_max_100,
                        "height": h_max_100,
                        "rectanglelabels": [label_name]
                    }
                }
            ],
            "score": float(rtdetr_box_data.conf[0])
        }
        
        with open(f"{dataset_base}/predictions/{json_name}", "w") as f:
            json.dump(label_studio_json, f, indent=2)
            
        print(f"💾 [方案 B 閉環] 成功打包預測 JSON 檔案：{json_name}")
    except Exception as e:
        print(f"⚠️ [方案 B 閉環] 打包預測 JSON 失敗: {e}")

# =========================================================================
# 🕸️ 4. 定義 LangGraph 節點 (Nodes)
# =========================================================================

def preprocess_node(state: AgentState) -> Dict[str, Any]:
    """前處理節點：解析 Kafka 事件，取得影像/影片，並跑本地 RT-DETR"""
    print("🤖 [Node: Preprocess] 正在解析 Kafka 原始事件與執行目標偵測...")
    event_data = state["event_data"]
    
    alert_type = event_data.get("event_type", "Pending_VLM_Review") 
    # env_clues 讀取邊緣 AI 的文字描述（vlm_summary），提供 VLM 更豐富的現場線索
    # 若無 vlm_summary，退而取用 event_type 作為基本線索
    env_clues = event_data.get("vlm_summary") or event_data.get("event_type", "No specific objects")
    
    dev_id = event_data.get("device_id")
    if dev_id is not None:
        target_device_id = int(dev_id)
        cam_id = f"Room_{target_device_id}"
    else:
        raw_cam_id = event_data.get("camera_id", "101")
        if "Room_" in str(raw_cam_id):
            parts = str(raw_cam_id).split('_')
            target_device_id = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 101
            cam_id = f"Room_{target_device_id}"
        else:
            target_device_id = int(raw_cam_id) if str(raw_cam_id).isdigit() else 101
            cam_id = f"Room_{target_device_id}"

    clip_path = event_data.get("clip_path", "/vids/fallback.mp4")
    image_filename = event_data.get("image_filename") or event_data.get("snapshot_path")
    base_dir = PROJECT_ROOT
    
    is_s3 = False
    if image_filename and image_filename.startswith("s3://"):
        img_path = image_filename
        is_s3 = True
    elif image_filename and os.path.isabs(image_filename):
        img_path = image_filename
    elif image_filename:
        img_path = os.path.join(base_dir, image_filename)
    else:
        img_path = os.path.join(base_dir, f"snapshot_{cam_id}.jpg")

    full_clip_path = os.path.join(base_dir, clip_path.lstrip("/")) if not os.path.isabs(clip_path) else clip_path
    frame_cv2 = None
    local_backup_img = ""

    if is_s3:
        try:
            _path_parts = img_path.replace("s3://", "").split("/", 1)
            _bucket = _path_parts[0]
            _key = _path_parts[1]
            s3_client = boto3.client('s3')
            s3_obj = s3_client.get_object(Bucket=_bucket, Key=_key)
            img_bytes = s3_obj['Body'].read()
            frame_cv2 = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            s3_filename = os.path.basename(img_path)
            local_backup_img = os.path.join(base_dir, "active_learning_dataset", "images", s3_filename)
            os.makedirs(os.path.dirname(local_backup_img), exist_ok=True)
            with open(local_backup_img, "wb") as f:
                f.write(img_bytes)
        except Exception as s3_read_err:
            print(f"❌ VLM 從 S3 抓取圖片失敗: {s3_read_err}")
            return {"should_send_report": False}
    else:
        if not os.path.exists(img_path):
            print(f"⚠️ 找不到邊緣截圖：{img_path}")
            return {"should_send_report": False}
        frame_cv2 = cv2.imread(img_path)

    input_source = frame_cv2 if is_s3 else img_path
    det_results = DETECTOR_MODEL(input_source, imgsz=640, verbose=False)[0]
    boxes = det_results.boxes

    highest_score = 0.0
    best_box = None
    if len(boxes) > 0:
        highest_score = float(boxes.conf.max())
        best_box_idx = boxes.conf.argmax()
        best_box = boxes[best_box_idx]

    return {
        "alert_type": alert_type,
        "env_clues": env_clues,
        "cam_id": cam_id,
        "clean_device_id": target_device_id,
        "clip_path": clip_path,
        "img_path": img_path,
        "local_backup_img": local_backup_img,
        "is_s3": is_s3,
        "highest_score": highest_score,
        "best_box": best_box,
        "should_send_report": True,
        "vlm_fall_reason_item": "unknown",
        "vlm_item_description": "無偵測到特定危險雜物"
    }

def bypass_node(state: AgentState) -> Dict[str, Any]:
    """快速過閘節點：高置信度事件不經 VLM 直接警報"""
    print("⚡ [Node: Bypass] RT-DETR 置信度極高，快速外發警報...")
    confidence_pct = f"{state['highest_score'] * 100:.1f}%"
    raw_report = f"【AI 快速通報】RT-DETR 高置信度解耦偵測 ({confidence_pct}) 觸發 {state['alert_type']} 事件。系統判定風險極高，已跳過 Video-LLM 複核，秒級推播警報！"
    severity = "high" if "fall" in state["alert_type"].lower() else "low"
    return {
        "raw_report": raw_report,
        "severity": severity
    }

def vlm_review_node(state: AgentState) -> Dict[str, Any]:
    """VLM 審查節點：呼叫時序 Video-LLM 進行二審"""
    print("🧠 [Node: VLM Review] 啟動 Qwen2.5-VL 影片/影像時序二審判定...")
    
    # 影像巡檢任務
    if state["alert_type"] == "Routine_Environment_Sanity_Check":
        prompt_text = (
            "You must reply ONLY in Traditional Chinese (繁體中文).\n"
            "You are an AI data curator helping to collect training data for a DETR (Bed and Wheelchair) detection model in a care center.\n"
            "Please carefully inspect the image and report if you can clearly see a 'bed' (病床) or a 'wheelchair' (輪椅).\n"
            "Please output a structured report using this exact template:\n\n"
            "【安養中心輪椅與床鋪標註收集報告】\n"
            f"1. 巡檢相機: {state['cam_id']}\n"
            "2. 發現目標: (輪椅 / 病床 / 皆無)\n"
            "3. 目標位置與狀態描述: (請提供非常詳細且豐富的描述，包含物品的具體擺放位置、周圍環境狀況，以及任何潛在的安全隱患，字數至少 50 字)\n\n"
            "==============================\n"
            "【主動探索學習模組】\n"
            "請在報告最尾端，嚴格且只以一組完整的 JSON 格式（不要包含 markdown 標籤或 ```json 字樣）"
            "輸出你看到畫面中的目標（'wheelchair' 或 'bed'，如皆無請輸出 'unknown'），格式必須完全對齊如下：\n"
            '{"item_name": "物品英文名", "description": "物品的中文描述"}'
        )
        vlm_input_source = [state["local_backup_img"] if (state["is_s3"] and state["local_backup_img"]) else state["img_path"]]
    else:
        # 動態影片審查任務
        clip_p = state["clip_path"]
        if clip_p.startswith("s3://") or clip_p.startswith("http://") or clip_p.startswith("https://"):
            full_clip_path = clip_p
        else:
            full_clip_path = os.path.join(PROJECT_ROOT, clip_p.lstrip("/")) if not os.path.isabs(clip_p) else clip_p
            
        if clip_p.startswith("s3://") or not os.path.exists(full_clip_path):
            print(f"⚠️ 警報影片位於 S3 雲端 ({clip_p})，自動降級使用 Snapshot 截圖二審...")
            vlm_input_source = [state["local_backup_img"] if (state["is_s3"] and state["local_backup_img"]) else state["img_path"]]
        else:
            vlm_input_source = extract_key_frames(full_clip_path, num_frames=5)
            
        prompt_text = (
            "You must reply ONLY in Traditional Chinese (繁體中文).\n"
            "You are a highly experienced AI head nurse and safety analyst in a smart elderly care center.\n"
            "Please carefully analyze the provided visual sequence to evaluate the person's dynamic safety.\n"
            "CRITICAL RULES FOR FALSE ALARMS & TRUE FALLS:\n"
            "- If the person is clearly walking safely, sitting properly on a chair, or standing safely, you MUST evaluate the situation as safe (Severity: Low).\n"
            "- If the person is lying on the floor, sprawling, or shows an abnormal posture on the ground, you MUST evaluate it as a true fall (Severity: High). Do not assume they are doing yoga or resting on the floor in a public space!\n"
            "- IMPORTANT: Pay special attention to Human-Object Interaction (HOI), such as how the person interacts with walkers, wheelchairs, furniture edges, or obstacles on the floor.\n"
            "- IMPORTANT: Attempt to predict intent (e.g., trying to get out of bed, losing balance) and explain the potential cause of the incident based on contextual clues.\n"
            f"Edge system clues: {state['env_clues']}.\n\n"
            "Please output a structured alert report using this exact template:\n\n"
            "【安養中心緊急通報（Video-LLM 原生影片二審版）】\n"
            "1. 現場狀況分析: (請提供極度詳細的分析，包含長者的肢體動作細節、移動軌跡、與周遭物品的互動關係（HOI），並推測其意圖或跌倒的前因後果，字數至少 100 字)\n"
            "2. 危險程度評估: (High / Medium / Low) - 請務必嚴格評估，若為安全動作請標示 Low\n"
            "3. 醫療建議行動: (請針對您推測的狀況，提供非常詳細且具體的第一線護理處置建議與避免二次傷害的注意事項，字數至少 50 字)\n\n"
            "==============================\n"
            "【主動探索學習模組】\n"
            "請在報告最尾端，嚴格且只以一組完整的 JSON 格式（不要包含 markdown 標籤或 ```json 字樣）"
            "輸出畫面中「最可能導致跌倒或值得注意的現場關鍵物（例如 slipper, wire, iv_pole, walker，如皆無請輸出 'unknown'）」，格式必須完全對齊如下：\n"
            '{"item_name": "物品英文名", "description": "物品的中文具體描述與現場狀況分析"}'
        )

    try:
        response = ollama.chat(
            model=VLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt_text, 'images': vlm_input_source}]
        )
        raw_report = response['message']['content'].strip()
        
        # 清理關鍵影格快取
        for f_temp in vlm_input_source:
            if "temp_vlm_frames" in f_temp and os.path.exists(f_temp):
                os.remove(f_temp)
                
        # 解析危險物品 JSON
        vlm_fall_reason_item = "unknown"
        vlm_item_description = "無偵測到特定危險雜物"
        try:
            json_match = re.search(r'\{"item_name".*?\}', raw_report, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted_item = json.loads(json_str)
                vlm_fall_reason_item = extracted_item.get("item_name", "unknown")
                vlm_item_description = extracted_item.get("description", "無偵測到特定危險雜物")
        except Exception as je:
            print(f"⚠️ 解析 VLM JSON 失敗: {je}")

        # 無論是否成功解析 JSON，一律將 "=======" 以後的所有文字（包含模型亂回傳的 prompt）通通切斷移除，確保前端乾淨
        if "==============================" in raw_report:
            raw_report = raw_report.split("==============================")[0].strip()
        elif "【主動探索學習模組】" in raw_report:
            raw_report = raw_report.split("【主動探索學習模組】")[0].strip()
        elif "請在報告最尾端" in raw_report:
            raw_report = raw_report.split("請在報告最尾端")[0].strip()

        severity = "high"
        severity_match = re.search(r"危險程度評估:\s*(High|Medium|Low)", raw_report, re.IGNORECASE)
        if severity_match:
            parsed_sev = severity_match.group(1).lower()
            if parsed_sev == "low":
                severity = "low"
        elif "low" in raw_report.lower() and "high" not in raw_report.lower():
            severity = "low"
        
        return {
            "raw_report": raw_report,
            "vlm_fall_reason_item": vlm_fall_reason_item,
            "vlm_item_description": vlm_item_description,
            "severity": severity
        }
    except Exception as e:
        print(f"❌ Ollama VLM 失敗: {e}")
        return {
            "raw_report": f"【系統二審異常】影像審查失敗。原因: {e}",
            "severity": "high"
        }

def active_learning_node(state: AgentState) -> Dict[str, Any]:
    """主動學習打包節點：僅精準封裝 LangGraph 判定的誤報 (false alarm) 與疑難樣本"""
    is_false_alarm = (state.get("severity") == "low")
    has_target_hazard = (state.get("vlm_fall_reason_item", "unknown") != "unknown")
    alert_type = state.get("alert_type", "")

    do_package = False
    target_category = "false_alarms"

    if alert_type == "fall":
        # 跌倒事件：只有誤報才打包給 YOLO 重訓
        if is_false_alarm:
            target_category = "false_alarms"
            print(f"💾 [Node: Active Learning] 🎯 捕獲跌倒誤報 (YOLO 負樣本)，正在打包至 [{target_category}] 供未來重訓...")
            do_package = True
        else:
            print("⏭️ [Node: Active Learning] 該事件為【真實跌倒證據】，無需重訓，跳過主動學習打包。")
    else:
        # 輪椅與病床收集事件：只要有發現目標，就打包給 DETR 重訓
        if has_target_hazard:
            target_category = "hazard_objects"
            print(f"💾 [Node: Active Learning] 🎯 發現目標 ({state.get('vlm_fall_reason_item')})，正在打包至 [{target_category}] 供未來 RT-DETR 專項重訓...")
            do_package = True
        else:
            print("⏭️ [Node: Active Learning] 畫面中未發現輪椅或床鋪，跳過主動學習打包。")

    if do_package and state.get("best_box") is not None:
        active_label = state["vlm_fall_reason_item"] if has_target_hazard else None
        package_active_learning_sample(
            state["img_path"], 
            state["cam_id"], 
            state["best_box"], 
            vlm_inferred_label=active_label, 
            local_backup_path=state["local_backup_img"],
            category=target_category
        )

    return {}


def discard_node(state: AgentState) -> Dict[str, Any]:
    """雜訊攔截節點：置信度過低直接過濾"""
    print(f"🛑 [Node: Discard] 雜訊攔截！RT-DETR 置信度過低 ({state['highest_score'] * 100:.1f}%)，已成功攔截過濾。")
    return {"should_send_report": False}

# =========================================================================
# 🔀 5. 路由決策函數 (Conditional Router)
# =========================================================================
def route_decision(state: AgentState) -> str:
    """根據 RT-DETR 置信度分數決定流程方向"""
    if not state.get("should_send_report", True):
        return END

    if state["alert_type"] == "Routine_Environment_Sanity_Check":
        return "vlm_review"
        
    if state["alert_type"] == "fall":
        # 送到 Kafka 的跌倒事件都是 YOLO 信心偏低需要複審的，所以一律進 VLM
        return "vlm_review"

    score = state["highest_score"]
    if score >= 0.75:
        return "bypass"
    elif 0.35 <= score < 0.75:
        return "vlm_review"
    else:
        return "discard"

# =========================================================================
# 🛠️ 6. 構建並編譯 LangGraph 狀態圖
# =========================================================================
workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("bypass", bypass_node)
workflow.add_node("vlm_review", vlm_review_node)
workflow.add_node("active_learning", active_learning_node)
workflow.add_node("discard", discard_node)

# 設定進入點
workflow.set_entry_point("preprocess")

# 設定條件分流路由
workflow.add_conditional_edges(
    "preprocess",
    route_decision,
    {
        "bypass": "bypass",
        "vlm_review": "vlm_review",
        "discard": "discard",
        END: END
    }
)

# 串接後續節點
workflow.add_edge("bypass", END)
workflow.add_edge("discard", END)

# VLM 審查後，流向主動學習，最後結束
workflow.add_edge("vlm_review", "active_learning")
workflow.add_edge("active_learning", END)

# 編譯 StateGraph
app = workflow.compile()
print("✅ [LangGraph Brain Compiler] 護理大腦狀態圖編譯成功！")

# =========================================================================
# 📥 7. 主 Kafka 監聽循環
# =========================================================================
if __name__ == "__main__":
    print("📦 連線地端 Kafka 流數據引擎...")
    consumer = KafkaConsumer(
        'nursing-home-alerts',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda v: json.loads(v.decode('utf-8')), 
        auto_offset_reset='latest',  
        group_id='vlm-brain-cluster'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print("🚀 [護理長 LangGraph Agent 上線] 監聽警報佇列中...")

    for message in consumer:
        event_data = message.value
        print(f"\n📥 收到新警報事件！正在啟動 LangGraph Agent 處理流...")
        
        # 初始化狀態並執行 Graph
        initial_state = {"event_data": event_data}
        result = app.invoke(initial_state)
        
        # 若需要外發報告，發送到第二級 Kafka 管道供前端顯示
        # 如果是誤報 (severity == "low")，則不發送以避免前端彈出警報
        should_send = result.get("should_send_report", True)
        if result.get("severity") == "low":
            print(f"🛑 [VLM 攔截] 判定為誤報 (Low Severity)，已攔截該警報，不再發送至前端！")
            should_send = False

        if should_send and result.get("raw_report") is not None:
            iso_detected_at = event_data.get("detected_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
            
            hazard_obj = result.get('vlm_fall_reason_item')
            if hazard_obj in (None, "unknown"):
                hazard_obj = None

            final_report = {
                "device_id": result["clean_device_id"],
                "event_type": result["alert_type"],
                "clip_path": result["clip_path"],
                "detected_at": iso_detected_at,
                "snapshot_path": result["img_path"],
                "yolo_score": result["highest_score"],
                "vlm_summary": result["raw_report"],
                "hazard_object": hazard_obj,
                "detected_objects": result.get("detected_objects"),
            }
            
            producer.send('processed-reports', value=final_report)
            item_display = result.get('vlm_fall_reason_item')
            if item_display == "unknown" or not item_display:
                item_display = "無危險障礙物 (環境安全)"
            print(f"📢 [Kafka 2] LangGraph Agent 二審報告外發成功！(危險雜物: {item_display})")