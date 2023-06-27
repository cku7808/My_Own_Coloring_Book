import os
import time
import cv2
import numpy as np

# sigma, low_threshold, high_threshold
TRAIN_PATH = r"data\np_train\org"
TRAIN_SAVE = r"data\np_train\edge"

TEST_PATH = r"data\np_test\org"
TEST_SAVE = r"data\np_test\edge"

VAL_PATH = r"data\np_val\org"
VAL_SAVE = r"data\np_val\edge"

def saturate_contrast1(p, num):
    pic = p.copy()
    pic = pic.astype('int64')
    pic = np.clip(pic*num, 0, 255)
    pic = pic.astype('uint8')
    return pic

for cur in ['train','test',"val"]:
    if cur == 'train':
        cur_path = TRAIN_PATH
        sv_path = TRAIN_SAVE
    elif cur == "test":
        cur_path = TEST_PATH
        sv_path = TEST_SAVE
    else:
        cur_path = VAL_PATH
        sv_path = VAL_SAVE

    for file in os.listdir(cur_path):
        FILE_NAME = file.split(".")[0]
        gray_img = cv2.imread(os.path.join(cur_path, file), cv2.IMREAD_GRAYSCALE)
        gray_img = saturate_contrast1(gray_img, 2) 
        threshold1 = 0
        threshold2 = 360
        edge_img = cv2.Canny(gray_img, threshold1, threshold2)
        cv2.imwrite(os.path.join(sv_path,FILE_NAME)+'.png', edge_img)