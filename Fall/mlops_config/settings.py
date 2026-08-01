from pydantic_settings import BaseSettings
from pathlib import Path

# 取 Fall 資料夾為 ROOT
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # ==========================
    # 核心路徑設定
    # ==========================
    PROJECT_ROOT: Path = BASE_DIR
    
    # ==========================
    # 特徵提取設定 (YOLO-Pose)
    # ==========================
    POSE_MODEL_PATH: Path = BASE_DIR / "yolo11s-pose.pt"
    CONF_THRES: float = 0.25
    SEQ_LENGTH: int = 30
    STRIDE: int = 5
    
    # ==========================
    # 分類器訓練設定 (Action Transformer)
    # ==========================
    BATCH_SIZE: int = 16
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-3
    INPUT_DIM: int = 34
    D_MODEL: int = 64
    NHEAD: int = 4
    NUM_LAYERS: int = 2
    
    # ==========================
    # 模型匯出與 Triton 設定
    # ==========================
    TRITON_MAX_BATCH_SIZE: int = 16
    ONNX_OPSET_VERSION: int = 14
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()
