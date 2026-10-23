# 清空並重建展示用資料。在專案根目錄執行: PYTHONPATH=. python scripts/reseed_demo_data.py
from backend.core.database import SessionLocal
from backend.core.models import Location, Device, DetectEvent, DetectEventReport
db = SessionLocal()

db.query(DetectEventReport).delete()
db.query(DetectEvent).delete()
db.query(Device).delete()
db.query(Location).delete()
db.commit()

import backend.init_db as init_db
init_db.seed_demo_data(db)
print("DB Locations and Devices successfully updated!")
