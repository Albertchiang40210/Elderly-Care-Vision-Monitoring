# 🏥 智慧病房 Fall Detection - Demo & 重訓終極操作手冊

這份手冊是為 **8/20 實際 Demo** 以及您平時**擴充資料重訓**所準備的必勝秘笈。只要照著這份清單輸入指令，您就能完美掌控整套系統。

---

## 🚀 場景一：8/20 實際 Demo 當天 (啟動系統)

要在評審或長官面前展示系統，您只需要把系統的核心引擎和畫面跑起來即可。請依序開啟終端機分頁執行：

### 1. 喚醒所有背景大腦 (Docker 容器)
這會一次把資料庫、Kafka 訊息佇列、Triton 推理伺服器、Label Studio 標註平台、ClearML 儀表板全部叫醒。
```bash
cd Fall/tools
docker-compose -f docker-compose-label.yml -f docker-compose-triton.yml -f docker-compose-clearml.yml -f docker-compose-kafka.yml up -d
cd ../..
```

### 2. 啟動後端 API 神經樞紐
接收 AI 傳來的跌倒警報，並儲存到資料庫。
```bash
cd backend
# 啟動虛擬環境 (如果有)
uvicorn main:app --reload --port 8000
```

### 3. 啟動前端 UI 展示儀表板
啟動讓長官看的精美監控畫面。
```bash
cd frontend
npm run dev
```

### 4. 啟動攝影機 AI 追蹤眼 (核心推理)
這支程式會打開鏡頭或讀取影片，並同時串聯三個 AI 模型開始抓人、抓骨架、判斷跌倒，最後發送 Discord 通知。
```bash
cd Fall
# 啟動虛擬環境
python tools/inference_test.py
```

---

## 🧠 場景二：平時的 MLOps 自動重訓 (讓 AI 變聰明)

我們的三個 AI 模型是**完全獨立**的！這代表您可以「哪裡不好就只練哪裡」。
**注意：以下所有指令，都請在專案根目錄 (`AIPE03`) 執行！**

### 🟢 模型 A：RT-DETR (負責抓人體邊界框)
**何時需要練它？** 當您發現 AI 常常抓不到畫面邊緣的人、或是誤把假人當真人時。
1. **餵資料**：把新的照片丟進 `Fall/label_studio_data/images`。
2. **AI 幫您寫作業 (預標註)**：
   ```bash
   python Fall/tools/inference_to_labelstudio_sdk.py
   ```
3. **人工批改 (審核)**：去 Label Studio (`http://localhost:8082`) 調整邊界框，按下 Submit。
4. **送進煉丹爐 (自動重訓 + 熱部署)**：
   ```bash
   python Fall/tools/clearml_train_pipeline.py
   ```

### 🟡 模型 B：YOLO-Pose (負責抓 17 個關節)
**何時需要練它？** 當您發現 AI 框到人了，但骨架亂飄、腳跟頭反過來時。
1. **餵資料**：把新的照片丟進 `Fall/label_studio_data/pose_images`。
2. **AI 幫您寫作業 (預標註)**：
   ```bash
   python Fall/tools/pose_to_labelstudio_sdk.py
   ```
3. **人工批改 (審核)**：去 Label Studio 微調錯位的關節點，按下 Submit。
4. **送進煉丹爐 (自動重訓 + 熱部署)**：
   ```bash
   python Fall/tools/clearml_pose_train_pipeline.py
   ```

### 🟣 模型 C：Action Transformer (負責判斷連續動作)
**何時需要練它？** 當骨架都很準，但 AI 卻把「安全蹲下(squat)」誤判成「危險跌倒(fall)」時。
1. **餵資料**：把短短幾秒的動作影片丟進 `Fall/label_studio_data/videos`。
2. **AI 幫您寫作業 (預標註)**：
   ```bash
   python Fall/tools/action_to_labelstudio_sdk.py
   ```
3. **人工批改 (審核)**：去 Label Studio 選擇正確的動作標籤 (fall / sitdown / squat 等)，按下 Submit。
4. **送進煉丹爐 (自動重訓 + 熱部署)**：
   ```bash
   python Fall/tools/clearml_action_train_pipeline.py
   ```

---
💡 **終極心法**：不管您練了哪一個模型，只要在終端機看到 **🎉 部署成功** 或聽到 Discord 發出捷報聲，您的監視器程式 (`inference_test.py`) 就會**在一秒內瞬間變聰明**，不需要重新開機！
