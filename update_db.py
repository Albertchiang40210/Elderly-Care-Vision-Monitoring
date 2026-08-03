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
