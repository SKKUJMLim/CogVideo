import cv2
import numpy as np

baseline = cv2.VideoCapture("outputs/honey_milk_baseline.mp4")
guided = cv2.VideoCapture("outputs/fire_water.mp4")

differences = []

while True:
    ok1, frame1 = baseline.read()
    ok2, frame2 = guided.read()

    if not ok1 or not ok2:
        break

    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    differences.append(diff.mean())

print("Compared frames:", len(differences))
print("Mean pixel difference:", np.mean(differences))
print("Maximum frame difference:", np.max(differences))