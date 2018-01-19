import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from io import BytesIO
from tensorflow.python.framework import ops

def unpickle(data_batch_1):
    import pickle
    with open(data_batch_1, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict 



# 이미지 배열을 그리는 함수, 그림을 그려서 창을 띄워 주는 함수
def showarray(a, fmt='jpeg'):
    # First make sure everything is between 0 and 255
    a = np.uint8(np.clip(a, 0, 1)*255)
    # Pick an in-memory format for image display
    f = BytesIO()
    # Create the in memory image
    PIL.Image.fromarray(a).save(f, fmt)
    # Show image
    plt.imshow(a)
    plt.show()


    
# Open image
# PIL : Python Image Library
img0 = PIL.Image.open('data_batch_1')
#img0 = unpickle('data_batch_1')
print('type(img0):', type(img0))

# np.float32 : 이미지 정보를 ndarray 형태로 변경
img0 = np.float32(img0)

# Show Original Image
showarray(img0/255.0)

# https://github.com/09rohanchopra/cifar10/blob/master/cifar10-basic.ipynb
# from helper import get_class_names, get_train_data, get_test_data, plot_images

