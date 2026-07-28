try:
    from backend.events.service import (
        DeviceNotFoundError,
        serialize_event,
        is_delivered,
        rebroadcast_event,
        watch_delivery,
        handle_incoming_event,
        operator_names,
    )
except ModuleNotFoundError:
    from events.service import (
        DeviceNotFoundError,
        serialize_event,
        is_delivered,
        rebroadcast_event,
        watch_delivery,
        handle_incoming_event,
        operator_names,
    )
