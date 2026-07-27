import os
import psycopg2

def check_database():
    # 讀取當前目錄下的 .env 檔案
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    config = {}
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    config[key.strip()] = val.strip().strip('"').strip("'")
    
    # 優先從 .env 或系統環境變數中讀取設定，不使用寫死的敏感憑證
    db_user = config.get('DB_USER') or os.environ.get('DB_USER')
    db_password = config.get('DB_PASSWORD') or os.environ.get('DB_PASSWORD')
    db_host = config.get('DB_HOST') or os.environ.get('DB_HOST')
    db_port = config.get('DB_PORT') or os.environ.get('DB_PORT', '5432')
    db_name = config.get('DB_NAME') or os.environ.get('DB_NAME')

    if not all([db_user, db_password, db_host, db_name]):
        print("❌ 錯誤：未在 .env 或環境變數中設定完整的資料庫連線資訊 (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)。")
        print("💡 請在當前目錄建立 .env 檔案並設定相關欄位，切勿將帳密寫死在程式碼中。")
        return

    print(f"🔍 嘗試連線至資料庫: {db_host}:{db_port}/{db_name}")
    print(f"👤 使用者: {db_user}")

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            connect_timeout=5
        )
        print("✅ 資料庫連線成功！")
    except Exception as e:
        print(f"⚠️ 使用 Port {db_port} 連線失敗: {e}")
        print("💡 嘗試換用 Port 5432 連線...")
        try:
            conn = psycopg2.connect(
                host=db_host,
                port="5432",
                user=db_user,
                password=db_password,
                database=db_name,
                connect_timeout=5
            )
            print("✅ 資料庫連線成功！(使用 Port 5432)")
        except Exception as e2:
            print(f"❌ 連線失敗: {e2}")
            return

    try:
        with conn.cursor() as cur:
            # 1. 查詢所有使用者定義的資料表
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE';
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            if not tables:
                print("⚠️ 資料庫中目前沒有任何資料表。")
                return
            
            print(f"\n📊 發現資料表數量: {len(tables)}")
            for table in tables:
                # 2. 統計各資料表的資料筆數
                cur.execute(f'SELECT COUNT(*) FROM "{table}";')
                count = cur.fetchone()[0]
                print(f"   - 資料表 [{table}]: 共 {count} 筆資料")
                
                # 3. 如果有資料，印出最近的 3 筆
                if count > 0:
                    try:
                        # 試著找出可能的排序欄位，如 id 或 created_at
                        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}';")
                        cols = [r[0] for r in cur.fetchall()]
                        order_by = ""
                        if "id" in cols:
                            order_by = "ORDER BY id DESC"
                        elif "created_at" in cols:
                            order_by = "ORDER BY created_at DESC"
                        elif "detected_at" in cols:
                            order_by = "ORDER BY detected_at DESC"
                        
                        cur.execute(f'SELECT * FROM "{table}" {order_by} LIMIT 3;')
                        rows = cur.fetchall()
                        print(f"     👉 最近的 {len(rows)} 筆資料:")
                        for row in rows:
                            print(f"        {row}")
                    except Exception as err:
                        print(f"     ⚠️ 無法讀取資料表內容: {err}")
    except Exception as e:
        print(f"❌ 查詢時發生錯誤: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    check_database()

# =========================================================================
# 💡 [檔案說明與核心職責]
# 「它是資料庫健檢與告警記錄驗證工具 (Database Inspection Tool)。」
# 本腳本用於快速檢查 PostgreSQL / AWS RDS 資料庫連線狀態：
# 1. 讀取 .env 憑證並透過 psycopg2 安全連線至數據庫。
# 2. 自動掃描所有資料表名稱（如 events, reports, users）。
# 3. 查詢並印出各資料表中最新 3 筆數據內容與欄位結構，方便開發除錯與驗證告警下沉成果。
# =========================================================================
