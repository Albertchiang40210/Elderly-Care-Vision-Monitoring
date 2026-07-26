import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 檔，讓 os.getenv() 可以抓到裡面的值

# 從 .env 把五個變數組合成一個連線字串
# 格式是 SQLAlchemy 規定的：驅動程式://帳號:密碼@主機:埠號/資料庫名稱
# psycopg2 是 Python 連 PostgreSQL 的驅動程式
import urllib.parse

db_pass_raw = os.getenv('DB_PASSWORD', '')
db_pass_encoded = urllib.parse.quote_plus(db_pass_raw, safe='') if db_pass_raw else ''

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('DB_USER')}:{db_pass_encoded}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

db_host = os.getenv('DB_HOST', '')
connect_args = {}
if "amazonaws.com" in db_host:
    connect_args = {
        "sslmode": "verify-full",
        "sslrootcert": "global-bundle.pem"
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args
)

# 每次要跟資料庫做事（查詢、新增），就會開一個 session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base() # 之後 models.py 裡定義的表格，都要繼承這個 Base，這樣 SQLAlchemy 才知道哪些是資料庫的表

# 一個工具函式，負責「開一個 session 給你用，用完自動關掉」，避免你忘記關閉連線造成資源浪費
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()