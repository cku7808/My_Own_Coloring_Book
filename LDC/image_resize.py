import cv2
import os

PATH = r"datasets\styletransfer"
SAVE_PATH = r"datasets\resized"
for file in os.listdir(PATH):
    file_name = file.split(".")[0]
    img = cv2.imread(os.path.join(PATH,file))

    img = cv2.resize(img,(512, 512)) # 1280 * 1024
    cv2.imwrite(os.path.join(SAVE_PATH,file),img)

