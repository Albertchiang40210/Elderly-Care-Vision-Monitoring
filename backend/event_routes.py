try:
    from backend.events.router import *
except ModuleNotFoundError:
    from events.router import *
