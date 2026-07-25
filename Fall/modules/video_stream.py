import cv2
import time
import logging

logger = logging.getLogger("RobustVideoCapture")

class RobustVideoCapture:
    """高可用性 RTSP 影像擷取封裝 (支援自動斷線重連與防卡死)"""
    
    def __init__(self, rtsp_url: str, retry_interval: float = 3.0, max_retries: int = 10):
        self.rtsp_url = rtsp_url
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.cap = None
        self._connect()

    def _connect(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        
        logger.info(f"📹 正在連接 RTSP 串流: {self.rtsp_url}...")
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """讀取最新影格，若遇斷訊自動進行背景重連"""
        if self.cap is None or not self.cap.isOpened():
            self._reconnect()

        ret, frame = self.cap.read() if self.cap else (False, None)
        
        if not ret or frame is None:
            logger.warn(f"⚠️ [RTSP 串流異常] 視訊讀取失敗，啟動自動重連機制 ({self.rtsp_url})...")
            return self._reconnect()

        return ret, frame

    def _reconnect(self):
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"🔄 重連嘗試 ({attempt}/{self.max_retries}) - {self.rtsp_url}...")
            self._connect()
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"✅ [RTSP 重連成功] 串流恢復正常！")
                    return ret, frame
            time.sleep(self.retry_interval)
            
        logger.error(f"❌ [RTSP 斷線超時] 無法重連至攝影機: {self.rtsp_url}")
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
