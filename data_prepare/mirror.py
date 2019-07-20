import os
import cv2
import numpy as np
from tqdm import tqdm

image_root = r'D:\work\job\face-landmark\dataset\train'
label_path = r'D:\work\job\face-landmark\dataset\label\train.txt'
save_root = r'D:\work\job\face-landmark\dataset\train_mirror'
save_mirror_rec = r'D:\work\job\face-landmark\dataset\label\train_mirror-re.csv'


def build_map():
    landmark = {}
    for index in range(17):
        landmark[index] = 16 - index

    for index in range(17, 27):
        landmark[index] = 43 - index

    for index in range(27, 31):
        landmark[index] = index

    for index in range(31, 36):
        landmark[index] = 66 - index

    landmark[36] = 45
    landmark[37] = 44
    landmark[38] = 43
    landmark[39] = 42
    landmark[40] = 47
    landmark[41] = 46
    for index in range(36, 42):
        landmark[landmark[index]] = index

    for index in range(48, 55):
        landmark[index] = 102 - index

    for index in range(55, 60):
        landmark[index] = 114 - index

    for index in range(60, 65):
        landmark[index] = 124 - index
        for index in range(65, 68):
            landmark[index] = 132 - index

    return landmark


font = cv2.FONT_HERSHEY_SIMPLEX
if not os.path.exists(save_root):
    os.mkdir(save_root)
with open(label_path) as f:
    landmark = build_map()
    # for index in range(68):
    # 	print('key: {}, value: {}'.format(index, landmark[index]))
    # exit(0)
    keypoint = {}
    f_mirror = open(save_mirror_rec, 'w')
    for line in tqdm(f.readlines()):
        line = line.strip()
        name, labels = line.split(',')
        tmp_path = os.path.join(image_root, name)
        assert os.path.exists(tmp_path)
        labels = labels.split(' ')
        img = cv2.imread(tmp_path)
        h, w, _ = img.shape
        for index in range(68):
            keypoint[index] = [labels[2 * index], labels[2 * index + 1]]
        # cv2.circle(img, (int(labels[2 * index]), int(labels[2 * index + 1])), 1, [0, 255, 0], -1)
        # cv2.circle(crop_img, (x_index, y_index), 1, [0, 0, 255], -1)
        # if index == 18:
        # cv2.putText(img, str(index), (int(labels[2 * index]), int(labels[2 * index + 1])), font, 0.5, (0, 255, 0), 1)
        img_x = cv2.flip(img, 1)
        keypoint_mirror = {}
        img_save_path = os.path.join(save_root, 'mirror_' + name)
        # cv2.imwrite(img_save_path, img_x)
        keypoint_line = ''
        for index in range(68):
            # x, y = (keypoint[index])
            # x, y = int(x), int(y)
            # flip_x = w - x
            key = landmark[index]
            keypoint_mirror[key] = [w - int(keypoint[index][0]), int(keypoint[index][1])]
        # keypoint_line = keypoint_line + ',' + str(keypoint_mirror[key][0]) + ',' + str(keypoint_mirror[key][1])

        for index in range(68):
            keypoint_line = keypoint_line + ',' + str(keypoint_mirror[index][0]) + ',' + str(keypoint_mirror[index][1])
        # keypoint_line = keypoint_line + ',' + str(keypoint_mirror[index][0]) + ',' + str(keypoint_mirror[index][1])
        # keypoint = keypoint + ',' + str(flip_x) + ',' + str(y)
        # cv2.circle(img_x, (flip_x, y), 2,  [255, 255, 0], 1)
        # cv2.putText(img_x, str(key), (flip_x, y), font, 0.5, (0, 255, 0), 1)
        # htitch= np.hstack((img, img_x))
        # cv2.imshow('demo-1', img_x)
        # cv2.waitKey(0)
        keypoint_line = 'mirror_' + name + ',' + keypoint_line
        f_mirror.write(keypoint_line)
        f_mirror.write('\n')
    f_mirror.close()
