# backend/event_service.py
# Root shim forwarding to backend.events.service
from backend.events.service import (
    DeviceNotFoundError,
    serialize_event,
    is_delivered,
    rebroadcast_event,
    watch_delivery,
    handle_incoming_event,
    operator_names,
)
