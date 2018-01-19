# Using TensorFlow for Deep Dream
#---------------------------------------
# From: Alexander Mordvintsev
#      --https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream
#
# Make sure to download the deep dream model here:
#   https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip


# inception5h: 딥드림을 시작하려면 구글 넷 파일을 먼저 다운로드 해야한다.
# 구글넷: cifar-1000을 대상으로 미리 학습해 놓은 CNN이다.
# cifar-1000: https://www.cs.toronto.edu/~kriz/cifar.html
# image-net

# tensorflow_inception_graph.pb : 구글넷

# Run:
#  me@computer:~$ wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip 
#  me@computer:~$ unzip inception5h.zip
#
#  More comments added inline.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from io import BytesIO
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

#os.chdir('/home/nick/Documents/tensorflow/inception-v1-model/')

# Model location
model_fn = 'tensorflow_inception_graph.pb'

# 그래프: 계산의 단위, tf.Operation의 셋트
# GraphDef: Graph 객체에서 계산을 수행해주는 가장 기본적인 토대
# FastGFile: 쓰레드 락킹없이 사용 가능한 파일 입출력 래퍼클래스
# 

# Load graph parameters
with tf.gfile.FastGFile(model_fn, 'rb') as f: #바이너리파일을 f라고 부름
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Create placeholder for input
t_input = tf.placeholder(np.float32, name='input')

# Imagenet average bias to subtract off images
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# Create a list of layers that we can refer to later
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]

# Count how many outputs for each layer
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# Print count of layers and outputs (features nodes)
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 30 # picking some feature channel to visualize

# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

# showarray 이미지 배열을 그리는 함수, 그림을 그려서 창을 띄어주는 함수
def showarray(a, fmt='jpeg'): #a가 배열
    # First make sure everything is between 0 and 255
    a = np.uint8(np.clip(a, 0, 1)*255) # clip함수가 범위사이의 숫자값으로 만들어줌
    # Pick an in-memory format for image display
    f = BytesIO()
    # Create the in memory image
    PIL.Image.fromarray(a).save(f, fmt)
    # Show image
    plt.imshow(a)
    plt.show()

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)


# The following function returns a function wrapper that will create the placeholder
# inputs of a specified dtype
# 명시된 타입에 따르는 placeholder를 생성해준다.

def tffunc(*argtypes): #*매개변수가 동적으로 들어올때 argtypes에 두개가 들어온다.
   
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))  #argtypes 수 만큼 placeholder생성
    
    # closure 함수외부에서 동적으로 접근할수있도록 해줌
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size): #이미지를 특정크기로 조정한다.
    img = tf.expand_dims(img, 0)
    # Change 'img' size by linear interpolation
    # resize_bilinear: 선형 이미지 보간 함수, 두지점간에 자연스러운 선을 만들어줌
    # 두줄 선형 보간법 사용하기
    return tf.image.resize_bilinear(img, size)[0,:,:,:]


# tiled 함수 이미지의 타일(tile)에 각 경사도를 계산해주는 함수 
# 쿡북 399쪽
def calc_grad_tiled(img, t_grad, tile_size=512): 
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    # Pick a subregion square size
    sz = tile_size
    # Get the image height and width
    h, w = img.shape[:2]
    # Get a random shift amount in the x and y direction
    sx, sy = np.random.randint(sz, size=2)
    # Randomly shift the image (roll image) in the x and y directions
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    # Initialize the while image gradient as zeros
    grad = np.zeros_like(img)
    # Now we loop through all the sub-tiles in the image
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            # Select the sub image tile
            sub = img_shift[y:y+sz,x:x+sz]
            # Calculate the gradient for the tile
            g = sess.run(t_grad, {t_input:sub})
            # Apply the gradient of the tile to the whole image gradient
            grad[y:y+sz,x:x+sz] = g
    # Return the gradient, undoing the roll operation
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

#render_deepdream 딥드림함수를 구현한다.
def render_deepdream(t_obj, img0=img_noise, 
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    # octave_n=4 정해진 수치만큼 for문장이 반복됨
    # 매개변수
    # t_obj: tensor객체
    # img0 : float32의 값을 가진 ndarray 객체
    
    # defining the optimization objective, the objective is the mean of the feature

    # 선택한 속성들의 평균값
    t_score = tf.reduce_mean(t_obj)
    # Our gradients will be defined as changing the t_input to get closer to
    # the values of t_score.  Here, t_score is the mean of the feature we select,
    # and t_input will be the image octave (starting with the last)
    # gradients : 미분계수구하기
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    

    # Store the image
    img = img0
    # Initialize the octave list
    octaves = [] # octaves 교재400쪽
    # Since we stored the image, we need to only calculate n-1 octaves
    for i in range(octave_n-1):
        # Extract the image shape
        # 이미지의 폭과 너비
        hw = img.shape[:2]
        # Resize the image, scale by the octave_scale (resize by linear interpolation)
        # 옥타브의 크기에 맞게 이미지 조절
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        # Residual is hi.  Where residual = image - (Resize lo to be hw-shape)
        # 고주파이미지
        hi = img-resize(lo, hw)
        # Save the lo image for re-iterating
        # 반복을 위하여 저주파 이미지 저장
        img = lo
        # Save the extracted hi-image
        octaves.append(hi) # 고주파 이미지를 리스트에 저장
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            # Start with the last octave
            hi = octaves[-octave]
            #
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            # Calculate gradient of the image.
            # 이미지 경사도 계산
            g = calc_grad_tiled(img, t_grad)
            # Ideally, we would just add the gradient, g, but
            # we want do a forward step size of it ('step'),
            # and divide it by the avg. norm of the gradient, so
            # we are adding a gradient of a certain size each step.
            # Also, to make sure we aren't dividing by zero, we add 1e-7.
            
            # 1e-7는 0으로 나누기 오류 방지를 위해서...
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')
        showarray(img/255.0)
        

# Run Deep Dream
if __name__=="__main__":
    # Create resize function that has a wrapper that creates specified placeholder types
    resize = tffunc(np.float32, np.int32)(resize)
    #resize는 변수고 tffunc는 함수
    # Open image
    # PIL: Python Image Library
    img0 = PIL.Image.open('./image/book_cover.jpg')
    print( type(img0 ))
    
    
    # np.float32: 이미지 정보를 ndarray 형태로 변경
    img0 = np.float32(img0)
    
    # Show Original Image
    showarray(img0/255.0)

    # Create deep dream
    render_deepdream(T(layer)[:,:,:,channel], img0, iter_n=15)

    sess.close()