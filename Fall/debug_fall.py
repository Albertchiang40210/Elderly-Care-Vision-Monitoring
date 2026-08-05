import cv2
import torch
import numpy as np
from modules.fall_detector import FallDetectorLogic
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
frame = cv2.imread("../backend/static/images/snapshot_cam_0.jpg") # if exists
# wait, I don't have the image. 

detector = FallDetectorLogic(img_w=1280, img_h=720)

kpts = []
for i in range(17): kpts.append([640, 100+i*30])

raw_box = [600, 80, 680, 700]
conf_val = 0.9

class DummyKeypoints:
    def __init__(self):
        self.xy = torch.tensor([[kpts]])

class DummyBoxes:
    def __init__(self):
        self.id = torch.tensor([1])
        self.conf = torch.tensor([0.9])
        self.xyxy = torch.tensor([raw_box])

class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()
        self.keypoints = DummyKeypoints()

out, trig = detector.process_inference_results([DummyResult()])
print(out)
