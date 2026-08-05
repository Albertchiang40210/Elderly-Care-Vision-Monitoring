import cv2
import numpy as np

# Simulate standing person coordinates
img_h, img_w = 720, 1280

# Standing person (x, y, conf)
kpts = [
    [640, 100, 0.9], # 0 nose
    [640, 100, 0.9], # 1
    [640, 100, 0.9], # 2
    [640, 100, 0.9], # 3
    [640, 100, 0.9], # 4
    [610, 150, 0.9], # 5 l_shoulder
    [670, 150, 0.9], # 6 r_shoulder
    [610, 250, 0.9], # 7
    [670, 250, 0.9], # 8
    [610, 350, 0.9], # 9
    [670, 350, 0.9], # 10
    [620, 380, 0.9], # 11 l_hip
    [660, 380, 0.9], # 12 r_hip
    [620, 520, 0.9], # 13 l_knee
    [660, 520, 0.9], # 14 r_knee
    [620, 680, 0.9], # 15 l_ankle
    [660, 680, 0.9], # 16 r_ankle
]
smooth_kpts = np.array(kpts)
smooth_box = [600, 80, 680, 700] # x1, y1, x2, y2

hip_leg_y_dist = None
l_hip, r_hip = smooth_kpts[11], smooth_kpts[12]
hip_ys = []
if len(l_hip) > 2 and l_hip[2] > 0.3: hip_ys.append(l_hip[1])
if len(r_hip) > 2 and r_hip[2] > 0.3: hip_ys.append(r_hip[1])

l_ankle, r_ankle = smooth_kpts[15], smooth_kpts[16]
leg_ys = []
if len(l_ankle) > 2 and l_ankle[2] > 0.3: leg_ys.append(l_ankle[1])
if len(r_ankle) > 2 and r_ankle[2] > 0.3: leg_ys.append(r_ankle[1])

if not leg_ys:
    l_knee, r_knee = smooth_kpts[13], smooth_kpts[14]
    if len(l_knee) > 2 and l_knee[2] > 0.3: leg_ys.append(l_knee[1])
    if len(r_knee) > 2 and r_knee[2] > 0.3: leg_ys.append(r_knee[1])

if hip_ys and leg_ys:
    hip_y = sum(hip_ys) / len(hip_ys)
    leg_y = sum(leg_ys) / len(leg_ys)
    bh = abs(float(smooth_box[3]) - float(smooth_box[1]))
    if bh > 0:
        hip_leg_y_dist = (leg_y - hip_y) / bh

print(f"hip_y: {hip_y}, leg_y: {leg_y}, bh: {bh}")
print(f"hip_leg_y_dist: {hip_leg_y_dist}")
