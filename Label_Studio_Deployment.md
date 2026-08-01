# Label Studio 啟動部屬與設定文件

本文檔說明如何使用 Docker Compose 部署並設定 Label Studio。

## 1. 系統需求
- Docker
- Docker Compose
- 至少 4GB RAM

## 2. Docker Compose 配置 (docker-compose.yml)

請在您的主機上建立一個目錄（例如 `label-studio`），並在該目錄下建立 `docker-compose.yml` 檔案，內容如下：

```yaml
version: "3.9"
services:
  app:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    volumes:
      - ./mydata:/label-studio/data
    environment:
      - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
      - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/images
    restart: unless-stopped
```

## 3. 目錄結構準備

在 `docker-compose.yml` 所在的目錄，建立資料儲存目錄：

```bash
mkdir -p mydata/images
```
這將確保 Label Studio 的資料會持久化儲存在 `mydata` 資料夾，並且允許讀取本地檔案。

## 4. 啟動服務

執行以下指令來在背景啟動 Label Studio：

```bash
docker-compose up -d
```

## 5. 初次設定

1. **登入系統**:
   打開瀏覽器，前往 `http://<您的伺服器IP>:8080`。初次進入時會要求建立管理員帳號與密碼。

2. **建立專案**:
   點選 "Create" 建立新專案，填寫專案名稱。

3. **設定標註介面 (Labeling Setup)**:
   在 "Labeling Setup" 選擇適合您任務的模板 (例如：Object Detection with Bounding Boxes)，然後配置您需要的類別標籤 (Labels)。

4. **匯入資料 (Import)**:
   - 您可以直接上傳檔案。
   - 或者使用 Cloud Storage / Local Storage 設定。如果您要讀取本機的 `mydata/images` 目錄，請到 Project Settings -> Cloud Storage -> Add Source Storage，選擇 Local Storage，路徑設定為 `/label-studio/data/images`。

## 6. 停止與更新

- **停止服務**:
  ```bash
  docker-compose down
  ```

- **更新版本**:
  ```bash
  docker-compose pull
  docker-compose up -d
  ```
