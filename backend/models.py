try:
    from backend.core.models import (
        User,
        Company,
        Location,
        Device,
        Staff,
        DetectEvent,
        DetectEventReport,
    )
except ModuleNotFoundError:
    from core.models import (
        User,
        Company,
        Location,
        Device,
        Staff,
        DetectEvent,
        DetectEventReport,
    )
