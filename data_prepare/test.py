import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\Think\Pictures\Thinkpad壁纸\d6a70f2442a7d9336dbcb217a44bd11373f00156.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_x = np.transpose(np.fliplr(np.transpose(img, (0, 2, 1))), (0, 2, 1))
# plt.imshow(img_x)
# plt.show()
cv2.imshow('demo-1', img_x)
cv2.waitKeyEx(0)