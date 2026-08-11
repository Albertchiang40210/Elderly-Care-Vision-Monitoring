from core.database import SessionLocal
from core.models import DetectEvent, DetectEventReport
db = SessionLocal()
try:
    print("Deleting DetectEventReport...")
    db.query(DetectEventReport).delete()
    print("Deleting DetectEvent...")
    db.query(DetectEvent).delete()
    db.commit()
    print("Success!")
except Exception as e:
    print("Failed to delete events:", e)
    db.rollback()
finally:
    db.close()
