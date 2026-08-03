from core.database import SessionLocal
from core.models import Location, Device
db = SessionLocal()

# 刪除所有現有資料，然後重新跑 init_db
db.query(Device).delete()
db.query(Location).delete()
db.commit()

import init_db
init_db.seed_demo_data(db)
print("DB Locations and Devices successfully updated!")
