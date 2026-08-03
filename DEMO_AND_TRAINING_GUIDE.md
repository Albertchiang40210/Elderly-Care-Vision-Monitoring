# 🏥 智慧病房 Fall Detection - Demo & 重訓終極操作手冊

這份手冊是為 **8/20 實際 Demo** 以及您平時**擴充資料重訓**所準備的必勝秘笈。只要照著這份清單輸入指令，您就能完美掌控整套系統。

---

## 🚀 場景一：8/20 實際 Demo 當天 (啟動系統)

要在評審或長官面前展示系統，您只需要把系統的核心引擎和畫面跑起來即可。請依序開啟終端機分頁執行：

### 1. 喚醒雲端 MLOps 後勤指揮中心 (基礎設施)
這會一次把資料庫、Kafka 訊息佇列、Label Studio 標註平台、ClearML 儀表板與所有的主動學習背景監聽程式全部叫醒。
```bash
./start_cloud.sh
```

### 2. 啟動邊緣端攝影機與戰情室 (核心推論與網頁 UI)
請開啟另一個終端機分頁，這會一鍵啟動前端戰情室、後端 API、串流中繼站以及 AI 推論引擎 (支援 GPU/CPU 自動切換)。
```bash
./start_edge.sh
```

### 3. 結束與關閉系統
當 Demo 結束時，只需執行以下指令即可安全關閉所有背景程序與 Docker 容器：
```bash
./stop_all.sh
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
