import os
import sys
import time
import cv2
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import threading
import json
from datetime import datetime  
import base64
import subprocess
import requests as _requests

# =========================================================================
# 🧭 自動修正 Python 模組搜尋路徑
# =========================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # 從 tools/ 往上推一層到 Fall/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import tritonclient.grpc as grpcclient  # 🚀 [Triton 整合] 引入 Triton gRPC 套件
except ImportError:
    grpcclient = None

from modules.fall_detector import get_body_angle_jit, PersonTrackerEMA, FallDetectorLogic
from modules.triton_parser import MockBox, MockResults, MockPoseKeypoints, MockPoseBoxes, MockPoseResults, parse_triton_yolo_pose

# =========================================================================
# 🔑 全域指定唯一合法 API KEY (物理強注，徹底解決 HTTP 401 拒絕問題)
# =========================================================================
VALID_API_KEY = "nAK4h8ARAJMjCSoWJ-uErx2KyZKGDF-jcXqmMUpkM_o"

def _get_valid_api_key():
    return VALID_API_KEY

# =========================================================================
# 🌟 專屬長照智慧模組（僅保留模組 G：環境安全巡檢）
# =========================================================================
try:
    from modules.sanity_check import RoutineSanityChecker  # 模組 G：VLM 閒置算力環境安全巡檢
    print("✅ [模組 G] 環境安全巡檢匯入成功")
except Exception as e:
    print(f"❌ [模組 G] 巡檢匯入失敗: {e}")
    RoutineSanityChecker = None

# =========================================================================
# 🛠️ MLOps 基礎建設：Kafka 初始化
# =========================================================================
from kafka import KafkaProducer
from ultralytics import YOLO, RTDETR

try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("✅ [Kafka] 訊息中心連線成功！雙向數據管線已就緒。")
except Exception as e:
    print(f"⚠️ [Kafka] 連線失敗: {e}")
    producer = None

# 🚀 開啟 GPU 硬體加速
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"🚀 推理引擎啟動，成功掛載 GPU 硬體加速裝置：{device}")

# =========================================================================
# 🛡️ 業界安防平滑追蹤器 (Box & Pose Smoothing Tracker)
# =========================================================================
# (Mock Class 與 Triton Parser 移至 modules.triton_parser)
# (PersonTrackerEMA 移至 modules.fall_detector)

# 初始化 Triton gRPC 連線
try:
    if grpcclient is not None:
        client_test = grpcclient.InferenceServerClient(url="localhost:8001")
        if client_test.is_server_live() and client_test.is_model_ready("yolo_pose"):
            triton_client = client_test
            print("✅ [Triton] 成功建立 gRPC 高速推論通道連線 (Port: 8001)！")
        else:
            print("🚀 [GPU 加速] 自動避開未就緒的 Triton 通道，直接啟動原生 Apple GPU (MPS) 超級加速引擎！")
            triton_client = None
    else:
        triton_client = None
except Exception as e:
    print("TRITON ERROR:", e)
    triton_client = None

# =========================================================================
# 🌟 全域載入模型 (自動尋找最新訓練出的 Active Learning 模型)
# =========================================================================
import glob

def get_latest_model(base_dir):
    search_pattern = os.path.join(base_dir, "*", "models", "best.pt")
    models = glob.glob(search_pattern)
    if not models:
        return None
    return max(models, key=os.path.getmtime)

# 1. 查找最新訓練出的 YOLO Pose 模型
custom_pose_dir = os.path.join(PROJECT_ROOT, "active_learning_pose_dataset", "models", "yolo_pose", "Fall_Detection")
custom_pose = get_latest_model(custom_pose_dir)
pose_model_name = custom_pose if custom_pose else "yolo11m-pose.pt"
print(f"🦴 [Model Loader] YOLO Pose 載入路徑: {pose_model_name}")

coreml_model_name = "yolo11s-pose.mlpackage"
# 如果使用客製化模型，則忽略 CoreML (因為目前沒有導出機制)
if not custom_pose and not os.path.exists(coreml_model_name) and os.path.exists(pose_model_name):
    try:
        print("🔄 首次載入：正在將 YOLO Pose 導出為 CoreML 格式以啟用 Mac 神經引擎加速...")
        from ultralytics import YOLO as YOLO_Exporter
        temp_model = YOLO_Exporter(pose_model_name)
        temp_model.export(format="coreml")
        print("🎉 CoreML 導出成功！")
    except Exception as e:
        print(f"⚠️ CoreML 導出失敗: {e}，將使用原生 CPU 模式。")
        
# YOLO model will be instantiated inside the worker to avoid thread-safety issues
# during concurrent model.fuse() calls.

output_frames = {}
frames_lock = threading.Lock()

# ─── 🎯 動態/即時畫框數據推送 ────────────────────
_BACKEND_DETECT_URL = "http://localhost:8010/events/live-detection"

def _push_detection(payload_dict: dict):
    try:
        resp = _requests.post(
            _BACKEND_DETECT_URL,
            json=payload_dict,
            headers={"X-API-Key": VALID_API_KEY, "Content-Type": "application/json"},
            timeout=0.5
        )
        if resp.status_code != 200:
            print(f"⚠️ [_push_detection Error] HTTP {resp.status_code}: {resp.text}")
    except Exception as err:
        print(f"⚠️ [_push_detection Exception] {err}")

import queue as _queue
_detection_queue: _queue.Queue = _queue.Queue(maxsize=10)

def _detection_queue_worker():
    while True:
        try:
            data = _detection_queue.get(timeout=1.0)
            _push_detection(data)
        except _queue.Empty:
            pass

_detection_bg_thread = threading.Thread(target=_detection_queue_worker, daemon=True)
_detection_bg_thread.start()

# =========================================================================
# 📹 核心：多鏡頭 Edge Worker (精簡純淨版：姿態跌倒 + 模組 G 環境巡檢)
# =========================================================================
def _async_process_video(frames, video_path, snapshot_path, cam_id, num_id, prod, export_fps):
    print(f"\n🎬 [{cam_id}] 異步背景合成前後 20 秒影片 ({len(frames)} 幀, 寫入幀率: {export_fps:.1f} FPS)...")
    try:
        if not frames: return
        frame_w, frame_h = 640, 360
        temp_raw_path = video_path.replace(".mp4", "_raw.mp4")
        # 先用 OpenCV 快速寫入 mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(temp_raw_path, fourcc, float(export_fps), (frame_w, frame_h))
        for f in frames:
            out.write(cv2.resize(f, (frame_w, frame_h)))
        out.release()
        
        # 再呼叫 FFmpeg 進行網頁標準化轉碼 (保證 Safari/iOS 絕對能播)
        import subprocess
        import os
        temp_final_path = video_path.replace(".mp4", "_final_temp.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_raw_path,
            "-c:v", "libx264", "-preset", "fast",
            "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            temp_final_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(temp_final_path):
            os.rename(temp_final_path, video_path)
        
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
            
        print(f"✅ [{cam_id}] 20 秒片段影片成功歸檔 (Web 完美相容)！")
    except Exception as async_err:
        print(f"❌ [{cam_id}] 背景處理影片失敗: {async_err}")

def camera_worker(camera_id, video_source):
    global producer, device, output_frames, frames_lock, triton_client, pose_model_name
    from ultralytics import YOLO, RTDETR
    
    # 每個執行緒擁有獨立的 YOLO 模型實例，避免多執行緒同時呼叫 fuse() 導致 KeyError/AttributeError
    print(f"🦴 [{camera_id}] 載入專屬 YOLO Pose 模型...")
    local_yolo_pose_model = YOLO(pose_model_name)
    try:
        local_yolo_pose_model.to(device)
    except Exception:
        pass

    # 2. 暫時使用原廠預設的 RT-DETR 模型 (因為目前缺乏醫院真實場景訓練資料)
    detr_model_path = os.path.join(PROJECT_ROOT, "rtdetr-l.pt")
    print(f"📦 [Model Loader] RT-DETR 載入路徑: {detr_model_path}")
    
    local_rtdetr_model = None
    try:
        if os.path.exists(detr_model_path):
            local_rtdetr_model = RTDETR(detr_model_path)
            local_rtdetr_model.to(device)
            print("✅ [RT-DETR] 環境危險物品辨識引擎啟動成功！")
    except Exception as e:
        print(f"⚠️ [{camera_id}] RT-DETR load error: {e}")

    # 🎯 檢查 video_source 是否為數字 (例如 "0" 或 "1")，如果是，則轉換為整數 (供 OpenCV 直接讀取本地實體/虛擬鏡頭)
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    print(f"🚀 鏡頭頻道 [{camera_id}] 啟動拉流：{video_source}")
    video_map = {
        "cam_0": "test1.MOV",
        "cam_1": "test2.MOV",
        "cam_2": "test3.MOV",
        "cam_3": "test4.MOV"
    }
    demo_video_name = video_map.get(camera_id, "test1.MOV")
    demo_video_path = os.path.join(PROJECT_ROOT, "Fall", "test_demo", demo_video_name)
    
    cap = None
    if isinstance(video_source, str) and video_source.startswith("rtsp://"):
        rtsp_url_ipv4 = video_source.replace("localhost", "127.0.0.1")
        gst_pipeline = (
            f"rtspsrc location={rtsp_url_ipv4} protocols=tcp latency=0 ! "
            f"rtph264depay ! h264parse ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        print(f"🚀 [{camera_id}] 正在嘗試點火 GStreamer 硬體解碼拉流...")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap is not None and cap.isOpened():
            ret_test, _ = cap.read()
            if not ret_test:
                cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            if cap is not None and cap.isOpened():
                ret_test, _ = cap.read()
                if not ret_test:
                    cap.release()
                    cap = None
    else:
        cap = cv2.VideoCapture(video_source)

    is_using_demo_video = False
    if cap is None or not cap.isOpened():
        if os.path.exists(demo_video_path):
            print(f"📱 [{camera_id}] 未偵測到實體相機，自動啟用備援：自動加載演示影片 ({demo_video_path}) 進行無縫循環推流！")
            cap = cv2.VideoCapture(demo_video_path)
            is_using_demo_video = True
        else:
            print(f"❌ 鏡頭頻道 [{camera_id}] 無法開啟影像源: {video_source} 且未找到演示影片")
            return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or fps < 24.0: fps = 30.0
    frame_delay = 1.0 / fps  

    PRE_SEC = 10
    POST_SEC = 10
    MAX_PRE_FRAMES = int(fps * PRE_SEC)
    MAX_POST_FRAMES = int(fps * POST_SEC)
    
    pre_video_buffer = deque(maxlen=MAX_PRE_FRAMES)
    post_video_buffer = []
    is_recording_post = False
    recording_fallen_id = None
    post_frame_count = 0

    frame_count = 0
    fps_calc_time = time.time()
    fps_calc_counter = 0
    measured_fps = 0.0
    
    # 相機位置設定：支援環境變數覆蓋，格式 CAM_META_cam_0="1:301 病房"
    # 預設值對應實體安裝位置，部署時可直接透過 .env 調整，不需改動程式碼
    _DEFAULT_CAM_META = {
        "cam_0": (1, "301 病房"),
        "cam_1": (2, "301 病房"),
        "cam_2": (3, "走廊"),
        "cam_3": (4, "交誼廳"),
    }
    _meta_str = os.environ.get(f"CAM_META_{camera_id}")
    if _meta_str and ":" in _meta_str:
        _parts = _meta_str.split(":", 1)
        numeric_id, location_str = int(_parts[0]), _parts[1]
    else:
        numeric_id, location_str = _DEFAULT_CAM_META.get(camera_id, (1, "301 病房"))
    
    results_pose = None
    last_annotated_frame = None
    vlm_report = "Waiting for alert..."
    last_alert_time = 0.0

    # 🎯 多人追蹤狀態與跌倒偵測邏輯處理器 (各鏡頭獨立)
    detector = None
    person_states = {}

    rtsp_writer_proc = None
    out_w, out_h = 1280, 720
    if camera_id == "Room_301_Bed":
        output_rtsp_url = "rtsp://localhost:8554/cam_out"
        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = cw if cw > 0 else 1280
        out_h = ch if ch > 0 else 720

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{out_w}x{out_h}', '-r', '24',
            '-i', '-', '-c:v', 'h264_videotoolbox', '-b:v', '4000k',
            '-realtime', 'true', '-g', '24', '-bf', '0',
            '-pix_fmt', 'yuv420p',
            '-f', 'rtsp', output_rtsp_url
        ]

        try:
            rtsp_writer_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            print(f"📡 [{camera_id}] 已成功啟動 MediaMTX Mac 硬體加速推流 ➔ {output_rtsp_url}")
        except Exception as e:
            print(f"⚠️ [{camera_id}] FFmpeg 推流啟動失敗: {e}")

    latest_push_frame = None
    latest_push_lock = threading.Lock()
    push_running = True
    detr_objects_list = []

    def ffmpeg_push_worker():
        nonlocal latest_push_frame, push_running
        target_delay = 1.0 / 24.0
        while push_running:
            t_p_start = time.time()
            frame_to_write = None
            with latest_push_lock:
                if latest_push_frame is not None:
                    frame_to_write = latest_push_frame.copy()

            if frame_to_write is not None and rtsp_writer_proc and rtsp_writer_proc.stdin:
                try:
                    rtsp_writer_proc.stdin.write(frame_to_write.tobytes())
                except Exception:
                    pass

            elapsed = time.time() - t_p_start
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    if rtsp_writer_proc:
        push_thread = threading.Thread(target=ffmpeg_push_worker, daemon=True)
        push_thread.start()

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        
        if not ret:
            if is_recording_post and 'final_video_path' in locals():
                is_recording_post = False
                full_20_sec_frames = list(pre_video_buffer) + list(post_video_buffer)
                actual_export_fps = measured_fps if measured_fps > 5.0 else 15.0
                threading.Thread(
                    target=_async_process_video,
                    args=(full_20_sec_frames, final_video_path, final_snapshot_path, camera_id, numeric_id, producer, actual_export_fps),
                    daemon=True
                ).start()
                
            if is_using_demo_video:
                break  # 影片結束，不要重播
            else:
                print(f"❌ [{camera_id}] 影像流已結束")
                break

        img_h, img_w, _ = frame.shape
        if img_w != 1280 or img_h != 720:
            frame = cv2.resize(frame, (1280, 720))
            img_h, img_w = 720, 1280

        frame_count += 1

        pose_was_updated = frame_count % 4 == 0 or results_pose is None
        if pose_was_updated:
            time.sleep(0.001)
            yolo_pose_success = False
            if triton_client is not None:
                # 暫無 Triton MOT
                pass
            
            if not yolo_pose_success:
                # 使用 ByteTrack 多人追蹤
                results_pose = local_yolo_pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.25, device=device)

            last_annotated_frame = frame.copy()

        # ─── 🎯【RT-DETR 環境物品偵測】───────────────────
        # 降頻執行，每 30 幀 (約 1 秒) 掃描一次，避免拖慢效能
        if frame_count % 30 == 0 and local_rtdetr_model is not None:
            try:
                # 提高信心門檻，減少誤判與雜訊
                results_detr = local_rtdetr_model.predict(frame, conf=0.45, verbose=False, device=device)
                detr_objects_list = []
                
                # 定義想要顯示的環境物品白名單 (過濾掉 person 以及細碎物品如手機、遙控器)
                HAZARD_WHITELIST = {
                    "chair", "couch", "bed", "dining table", "tv", "laptop", 
                    "refrigerator", "potted plant", "wheelchair"
                }

                if results_detr and len(results_detr) > 0:
                    r = results_detr[0]
                    names = r.names
                    for i, box in enumerate(r.boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = names.get(cls_id, "object")
                        
                        # 過濾：只留下白名單內的物品
                        if name not in HAZARD_WHITELIST:
                            continue

                        # 取得相對於寬高的 0-1 座標
                        xywhn = box.xywhn[0].tolist()
                        # xywh 轉 x1 y1 x2 y2 (0-1)
                        nx_c, ny_c, nw, nh = xywhn
                        nx1 = nx_c - nw / 2
                        ny1 = ny_c - nh / 2
                        nx2 = nx_c + nw / 2
                        ny2 = ny_c + nh / 2
                        
                        # [修復] 過濾掉與人類重疊的假陽性 (防止專屬模型把人誤認為病床)
                        is_overlapping_person = False
                        if results_pose and len(results_pose) > 0 and results_pose[0].boxes is not None:
                            for p_box in results_pose[0].boxes.xyxyn.cpu().numpy():
                                px1, py1, px2, py2 = p_box
                                # 簡單的 IoU 或包含測試
                                inter_x1 = max(nx1, px1)
                                inter_y1 = max(ny1, py1)
                                inter_x2 = min(nx2, px2)
                                inter_y2 = min(ny2, py2)
                                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                                detr_area = (nx2 - nx1) * (ny2 - ny1)
                                if detr_area > 0 and (inter_area / detr_area) > 0.4:  # 重疊超過 40% 就視為同一個物體(人)
                                    is_overlapping_person = True
                                    break
                        
                        if is_overlapping_person:
                            continue
                        
                        detr_objects_list.append({
                            "class": name,
                            "conf": round(conf, 2),
                            "bbox": [nx1, ny1, nx2, ny2]
                        })
            except Exception as e:
                pass


        annotated_frame = last_annotated_frame.copy() if last_annotated_frame is not None else frame.copy()

        # ─── 🎯【多目標追蹤與跌倒判定】───────────────────
        # 初始化分離的業務邏輯處理器
        if detector is None:
            detector = FallDetectorLogic(img_w=img_w, img_h=img_h)
            
        persons_out_list, any_fall_triggered_this_frame = detector.process_inference_results(results_pose, pose_was_updated)
        
        for p_data in persons_out_list:
            track_id = p_data["id"]
            active_conf = p_data["conf"]
            should_trigger_fall = p_data["is_fall"]
            smooth_box = p_data.get("smooth_box")
            smooth_kpts = p_data.get("smooth_kpts")
            
            # 繪製單人邊框與骨架 (已移除 OpenCV 原生繪製，全權交給前端 Canvas 繪製以避免重疊與座標錯位)
            # 只針對跌倒者畫紅框，其他人不畫框，保持畫面乾淨
            if smooth_box is not None:
                try:
                    bx1, by1 = int(float(smooth_box[0])), int(float(smooth_box[1]))
                    bx2, by2 = int(float(smooth_box[2])), int(float(smooth_box[3]))
                    
                    # 如果這是在錄影期間，而且這正是觸發警報的那個人；或是剛好判定為跌倒的瞬間
                    if (is_recording_post and track_id == recording_fallen_id) or should_trigger_fall:
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 4) # 粗紅框
                        p_label = f"ID:{track_id} {active_conf:.2f}"
                        cv2.putText(annotated_frame, p_label, (bx1, max(by1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                    
            if p_data.get("recovered"):
                print(f"ℹ️ [{camera_id}] (ID:{track_id}) 偵測到長者已自行站起，重置狀態！")
                vlm_report = f"ID:{track_id} Self-Recovered"
                
            # ⚡ 跌倒通報邏輯 (獨立針對每個 track_id)
            if p_data.get("new_fall_trigger"):
                # 計算中心點
                cx, cy = 0, 0
                if smooth_box is not None:
                    try:
                        cx = (float(smooth_box[0]) + float(smooth_box[2])) / 2
                        cy = (float(smooth_box[1]) + float(smooth_box[3])) / 2
                    except Exception:
                        pass
                
                # 鏡頭全局冷卻檢查 (30秒內同一支鏡頭只發送一次警報)
                current_time = time.time()
                is_cooldown = (current_time - last_alert_time <= 30.0)
                        
                if not is_cooldown:
                    last_alert_time = current_time
                    if not is_recording_post:
                        is_recording_post = True
                        recording_fallen_id = track_id
                        post_frame_count = 0
                        post_video_buffer = []
                        
                        vlm_save_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "backend", "static", "images")
                        os.makedirs(vlm_save_dir, exist_ok=True)
                        current_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        video_name = f"fall_clip_{camera_id}_{current_time_str}.mp4"
                        final_video_path = os.path.join(vlm_save_dir, video_name)
                    else:
                        if 'current_time_str' not in locals():
                            current_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        if 'final_video_path' not in locals():
                            video_name = f"fall_clip_{camera_id}_{current_time_str}.mp4"
                            vlm_save_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "backend", "static", "images")
                            final_video_path = os.path.join(vlm_save_dir, video_name)

                    snapshot_name = f"snapshot_{camera_id}_ID{track_id}_{current_time_str}.jpg"
                    final_snapshot_path = os.path.join(vlm_save_dir, snapshot_name)
                    cv2.imwrite(final_snapshot_path, annotated_frame)
                    
                    local_snap_url = f"/images/{os.path.basename(final_snapshot_path)}"
                    local_vid_url = f"/images/{os.path.basename(final_video_path)}"
                    
                    instant_payload = {
                        "device_id": numeric_id,
                        "camera_id": camera_id,
                        "location": location_str,
                        "event_type": "fall",
                        "person_id": track_id,
                        "clip_path": local_vid_url,
                        "detected_at": datetime.now().isoformat(),
                        "snapshot_path": local_snap_url,
                        "image_filename": final_snapshot_path,
                        "yolo_score": round(float(active_conf), 2),
                        "vlm_summary": f"【緊急通報】邊緣 AI 即時偵測到長者 (ID:{track_id}) 跌倒！請護理人員手動處置。"
                    }

                    if active_conf >= 0.4:
                        try:
                            headers = {"X-API-Key": VALID_API_KEY, "Content-Type": "application/json"}
                            res = _requests.post("http://localhost:8010/events", json=instant_payload, headers=headers, timeout=2.0)
                            if res.status_code in [200, 201]:
                                print(f"⚡ [{camera_id}] (ID:{track_id}) 【秒級即時告警】跌倒通知已 0 延遲轟入後端！")
                        except Exception:
                            pass
                        
                        # 讓高信心警報也進入 Kafka 進行 VLM 分析，以取得詳細報告
                        if producer is not None:
                            try:
                                producer.send('nursing-home-alerts', value=instant_payload)
                                producer.flush()
                            except Exception:
                                pass
                    else:
                        if producer is not None:
                            instant_payload["vlm_summary"] = None
                            instant_payload["status"] = "PENDING_VLM_ROUTE"
                            try:
                                producer.send('nursing-home-alerts', value=instant_payload)
                                producer.flush()
                                print(f"🧠 [{camera_id}] (ID:{track_id}) 跌倒信心偏低，送交 VLM 二次判斷！")
                            except Exception:
                                pass

        # 發送繪圖 JSON 給前端
        if persons_out_list or detr_objects_list or frame_count % 10 == 0:
            try:
                _detection_queue.put_nowait({
                    camera_id: {
                        "persons": persons_out_list,
                        "objects": detr_objects_list
                    },
                    "backend_fps": round(measured_fps, 2)
                })
            except Exception:
                pass

        if any_fall_triggered_this_frame:
            cv2.rectangle(annotated_frame, (0, 0), (img_w, img_h), (0, 0, 255), 12)
            cv2.putText(annotated_frame, "FALL DETECTED!", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(annotated_frame, "Normal", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # 將最終帶有標註的畫面存入 Buffer，供回放影片使用
        pre_video_buffer.append(annotated_frame)
        if is_recording_post:
            post_video_buffer.append(annotated_frame)
            post_frame_count += 1

        # 背景影片寫入邏輯
        if is_recording_post and post_frame_count >= MAX_POST_FRAMES:
            is_recording_post = False
            recording_fallen_id = None
            full_20_sec_frames = list(pre_video_buffer) + post_video_buffer
            

            if 'final_video_path' in locals():
                actual_export_fps = measured_fps if measured_fps > 5.0 else 15.0
                
                threading.Thread(
                    target=_async_process_video,
                    args=(full_20_sec_frames, final_video_path, final_snapshot_path, camera_id, numeric_id, producer, actual_export_fps),
                    daemon=True
                ).start()

        now_t = time.time()
        fps_calc_counter += 1
        if now_t - fps_calc_time >= 1.0:
            measured_fps = fps_calc_counter / (now_t - fps_calc_time)
            fps_calc_counter = 0
            fps_calc_time = now_t

        fps_text = f"FPS: {measured_fps:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        box_x1 = img_w - text_w - 20
        box_y1 = img_h - text_h - 20
        box_x2 = img_w
        box_y2 = img_h
        cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.putText(annotated_frame, fps_text, (box_x1 + 10, box_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"VLM Status: {vlm_report}", (40, img_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        small_frame = cv2.resize(annotated_frame, (320, 240))
        with frames_lock: output_frames[camera_id] = small_frame.copy()

        if rtsp_writer_proc:
            push_f = cv2.resize(annotated_frame, (out_w, out_h)) if (annotated_frame.shape[1] != out_w or annotated_frame.shape[0] != out_h) else annotated_frame
            with latest_push_lock:
                latest_push_frame = push_f.copy()

        t_elapsed = time.time() - t_start
        t_sleep = frame_delay - t_elapsed
        if t_sleep > 0: time.sleep(t_sleep)

    cap.release()

# =========================================================================
# 🏢 主執行緒控制
# =========================================================================
if __name__ == "__main__":
    camera_channels = {
        "cam_0": os.environ.get("CAM0_URL", "rtsp://localhost:8554/cam_0"),
        "cam_1": os.environ.get("CAM1_URL", "rtsp://localhost:8554/cam_1"),
        "cam_2": os.environ.get("CAM2_URL", "rtsp://localhost:8554/cam_2"),
        "cam_3": os.environ.get("CAM3_URL", "rtsp://localhost:8554/cam_3"),
    }

    print(f"🎬 全連鎖安養中心多鏡頭智能管線全面啟動...")
    
    threads = []
    for cam_id, stream_src in camera_channels.items():
        t = threading.Thread(target=camera_worker, args=(cam_id, stream_src))
        t.daemon = True; threads.append(t); t.start()
        
    headless_mode = "--headless" in sys.argv or os.environ.get("HEADLESS") == "1"
    
    try:
        if headless_mode:
            print("🖥️  [Headless] 以背景無 GUI 模式啟動...")
            while True:
                time.sleep(1.0)
        else:
            frame_interval = 1.0 / 30.0; window_positions = {}  
            while True:
                start_time = time.time(); active_windows = False
                with frames_lock: current_display_frames = output_frames.copy()
                    
                for idx, (cam_id, img_to_show) in enumerate(current_display_frames.items()):
                    if img_to_show is not None:
                        win_name = f"Fall Detection System - {cam_id}"
                        cv2.imshow(win_name, img_to_show)
                        if win_name not in window_positions:
                            x_pos = 40 + (idx * 335); y_pos = 80
                            cv2.moveWindow(win_name, x_pos, y_pos)
                            window_positions[win_name] = (x_pos, y_pos)
                        active_windows = True
                
                if active_windows:
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                sleep_time = frame_interval - (time.time() - start_time)
                time.sleep(sleep_time if sleep_time > 0 else 0.001)
                
    except KeyboardInterrupt: pass
    finally:
        if not headless_mode:
            cv2.destroyAllWindows()
        if producer is not None: producer.close()
