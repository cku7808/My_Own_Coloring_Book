import cv2
import numpy as np
from helper import *
from collections import deque
import tensorflow as tf
import os
from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import re
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.util import invert

def canny(image):
  # 엣지 감지 수행
  threshold1 = 100  # 첫 번째 임계값
  threshold2 = 200  # 두 번째 임계값
  edges = cv2.Canny(image, threshold1, threshold2)
  edges = invert(edges)
  return edges

def get_train(x_train_folder_path,y_train_folder_path):
    x_train_file_names = os.listdir(x_train_folder_path)
    y_train_file_names = os.listdir(y_train_folder_path)
    x_train = deque([])
    y_train = deque([])
    new_size = (512, 512)
    print(len(y_train_file_names), len(x_train_file_names))
    for i in x_train_file_names:
        from_mat = cv2.imread(x_train_folder_path + i, cv2.IMREAD_GRAYSCALE)
        from_mat = cv2.resize(from_mat, new_size)
        x_train.append(from_mat)


    for i in y_train_file_names:
        from_mat = cv2.imread(y_train_folder_path + i, cv2.IMREAD_GRAYSCALE)
        from_mat = cv2.resize(from_mat, new_size)
        from_mat = canny(from_mat)
        y_train.append(from_mat)

    return x_train,y_train


def get():
    y_train_folder_path = "C:/Users/qkrwp/sketchKeras/data/x_train" # 폴더 경로를 해당하는 경로로 변경해주세요
    x_train_folder_path = "C:/Users/qkrwp/sketchKeras/data/x_train"
    x_train, y_train = get_train(x_train_folder_path,y_train_folder_path )

    # 배열의 형태 변경
    x_train = np.expand_dims(x_train, axis=0)
    y_train = np.expand_dims(y_train, axis=0)
    print(x_train.shape, y_train.shape)

    x_train = x_train.reshape(4277, 512, 512, 1)  # 텐서의 형태 확인
    y_train = y_train.reshape(4277, 512, 512, 1)  # 텐서의 형태 확인
    model.compile(optimizer="Adam",
                loss="mae",
                metrics=["accuracy"])

    # 저장할 모델 파일 경로와 이름 설정
    checkpoint_path = 'model_checkpoint.h5'

    # ModelCheckpoint 콜백 생성
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='loss', save_weights_only=False,
                                          save_freq='epoch')
    history = model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=1, callbacks=[checkpoint_callback])


    return history

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 제한 설정 (여기에서는 4GB로 제한)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096*3)]
        )
    except RuntimeError as e:
        print(e)

with tf.device('/gpu:0'):
    model = load_model('mod.h5')
    print(device_lib.list_local_devices())
    get()