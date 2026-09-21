from ultralytics import RTDETR
import os
model = RTDETR('rtdetr-l.pt')
print(getattr(model, 'trainer', 'No trainer'))
