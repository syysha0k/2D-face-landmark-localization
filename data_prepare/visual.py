import os
import cv2

image_root = r'D:\work\job\face-landmark\dataset\train'
label_path = r'D:\work\job\face-landmark\dataset\landmark_result\record-199-normal-little.txt'

font = cv2.FONT_HERSHEY_SIMPLEX
i = 0
with open(label_path) as f:
    keypoint = {}
    for line in f.readlines():
        line = line.strip()
        name, labels = line.split(',')
        tmp_path = os.path.join(image_root, name)
        assert os.path.exists(tmp_path)
        labels = labels.split(' ')
        img = cv2.imread(tmp_path)
        img_resize = cv2.resize(img, (224, 224))
        h, w, _ = img.shape
        for index in range(68):
            keypoint[index] = [labels[2 * index], labels[2 * index + 1]]
            cv2.circle(img_resize, (int(float(labels[2 * index])), int(float(labels[2 * index + 1]))), 1, [0, 0, 255], -1)
        cv2.imwrite(name, img_resize)

