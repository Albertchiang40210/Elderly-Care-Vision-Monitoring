# ClearML 啟動部屬與設定文件

本文檔說明如何使用 Docker Compose 部署並設定 ClearML 伺服器 (ClearML Server)。

## 1. 系統需求
- Docker
- Docker Compose
- 至少 8GB RAM，建議 16GB 以上（用於 Elasticsearch, Redis, MongoDB）
- Linux 環境 (Ubuntu 20.04/22.04 建議)

## 2. 獲取官方部署配置

ClearML 官方維護了完整的 Docker 部署腳本，建議直接使用官方提供的 `docker-compose`。

```bash
# 建立目錄
mkdir -p /opt/clearml
cd /opt/clearml

# 下載官方 docker-compose.yml 
curl -L -O https://raw.githubusercontent.com/allegroai/clearml-server/master/docker/docker-compose.yml

# 下載環境變數設定檔
curl -L -O https://raw.githubusercontent.com/allegroai/clearml-server/master/docker/.env
```

## 3. 設定環境變數 (.env)

編輯剛剛下載的 `.env` 檔案以符合您的需求，常見的重要設定包含：

- `CLEARML_HOST_IP`：若要從外部存取，請設定為您伺服器的對外 IP。
- `CLEARML_ELASTIC_MEMORY`：可根據主機記憶體大小調整（預設通常夠用）。

若是在本機測試，通常保持預設即可（預設會綁定到 localhost 及預設 port）。

預設服務與 Port：
- Web UI: 8080
- API Server: 8008
- File Server: 8081

## 4. 啟動 ClearML 伺服器

執行以下指令啟動所有核心服務：

```bash
docker-compose up -d
```
> **注意**: 初次啟動會需要下載多個映像檔 (Elasticsearch, MongoDB, Redis, Web, API 等)，並且 Elasticsearch 初始化可能需要數分鐘。

## 5. 驗證與初次設定

1. **登入 Web UI**:
   打開瀏覽器，前往 `http://<您的伺服器IP>:8080`。

2. **建立 Credentials**:
   第一次進入時，到右上角的 Settings -> Workspace 頁面。點選 **"Create new credentials"** 來產生新的訪問金鑰 (Access Key) 與秘密金鑰 (Secret Key)。

## 6. 用戶端 (Client) 設定

要在您的 Python 訓練環境中連接這台 ClearML 伺服器，請在用戶端機器上進行設定：

```bash
# 安裝 clearml
pip install clearml

# 初始化設定
clearml-init
```
執行 `clearml-init` 後，系統會提示您輸入剛剛在 Web UI 產生的配置文字（包含 API, Web, Files 的 URL，以及 Access Key 和 Secret Key）。複製貼上後，用戶端即設定完成。

## 7. 停止與維護

- **停止服務**:
  ```bash
  docker-compose down
  ```
- 資料預設會透過 Docker Volumes 儲存在 `/opt/clearml/data` (依據 docker-compose.yml 內定義的 volume 設定)，請定期備份。
