import numpy as np
import pandas as pd
import os
import cv2

csv_path = r'D:\work\job\face-landmark\dataset\label\train_mirror-rex.csv'
image_path = r'D:\work\job\face-landmark\dataset\train_mirror'
key_pts = pd.read_csv(csv_path)

font = cv2.FONT_HERSHEY_SIMPLEX

for index in range(len(key_pts)):
    name = key_pts.iloc[index, 0]
    img = os.path.join(image_path, name)
    image = cv2.imread(img)
    key_points = key_pts.iloc[index, 1:].values
    for i in range(68):
        x = key_points[2 * i]
        y = key_points[2 * i + 1]
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(image, str(i), (x, y), font, 0.3, (0, 255, 0), 1)
    cv2.imshow('demo-1', image)
    cv2.waitKey(0)
    # exit(0)

