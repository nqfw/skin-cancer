import cv2
import numpy as np
import os
import glob

search_dirs = [
    r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_images_part_1",
    r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\raw",
    r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\test images"
]

results = []

for d in search_dirs:
    if not os.path.exists(d): continue
    for f in glob.glob(os.path.join(d, "*.jpg")):
        img = cv2.imread(f)
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([140, 255, 255]))
        c = cv2.countNonZero(mask)
        if c > 1000:
            results.append((c, f))

# Sort by pixel count descending
results.sort(reverse=True)
for count, filepath in results[:10]:
    print(f"{count} pixels -> {filepath}")
