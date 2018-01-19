
# Using TensorFlow for Stylenet/NeuralStyle
#---------------------------------------
#
# We use two images, an original image and a style image
# and try to make the original image in the style of the style image.
#
# Reference paper:
# https://arxiv.org/abs/1508.06576
#
# Need to download the model 'imagenet-vgg-verydee-19.mat' from:
#   http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

# 'imagenet-vgg-verydee-19.mat'다운받아서, 해당파일을 현재폴더에 붙여넣으시요.

# style-net
# 스타일 이미지를 학습하여 원본 이미지를 스타일 이미지의 스타일로 적용시키는 기법


import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()


# Image Files
# ./는 현재폴더 ../상위폴더
original_image_file = './image/book_cover.jpg' # 원본 파일 
style_image_file = './image/starry_night.jpg' # 스타일 파일

# Saved VGG Network path under the current project dir.
# imagenet-vgg-19 신경망
# vgg_path: 사전에 미리학습했던 신경망 파일
vgg_path = 'imagenet-vgg-verydeep-19.mat'
# .mat 바이너리 파일 학습했던 내용들을 저장해둔다.
# mat 파일 --> scipy --> python


# Default Arguments
original_image_weight = 5.0 # 원본 이미지의 최종 가중치
style_image_weight = 500.0 # 스타일 이미지의 최종 가중치
regularization_weight = 100
learning_rate = 0.001 # 학습율
generations = 1000 # 총 학습 횟수
#output_generations = 250
output_generations = 20 # 20회마다 출력하겠다.

#아담옵티마이저 관련변수
beta1 = 0.9 #1번째 moment에 대한 지수적 감쇠기
beta2 = 0.999 #2번째 moment에 대한 지수적 감쇠기

# Read in images, scipy를 이용한 이미지 로딩
original_image = scipy.misc.imread(original_image_file)
style_image = scipy.misc.imread(style_image_file)

# Get shape of target and make the style image the same
target_shape = original_image.shape

# imresize: 원본 이미지의 크기와 동일하게 스타일 이미지를 리사이징
style_image = scipy.misc.imresize(style_image, target_shape[1] / style_image.shape[1])

# VGG-19 Layer Setup
# From paper
# 문자열이 들어있는 리스트. 논문저자의 명명법에 따라 다음과 같이 layer를 정의한다.
# 첫글자가 c면 conv r이면 relu p면 maxpooling한다. 
# len(vgg_layers)는 36이다.

vgg_layers = ['conv1_1', 'relu1_1',
              'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1',
              'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1',
              'conv3_2', 'relu3_2',
              'conv3_3', 'relu3_3',
              'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1',
              'conv4_2', 'relu4_2',
              'conv4_3', 'relu4_3',
              'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1',
              'conv5_2', 'relu5_2',
              'conv5_3', 'relu5_3',
              'conv5_4', 'relu5_4']

# Extract weights and matrix means
# mat파일에서 매개변수를 추출해주는함수
# 튜플로 리턴
def extract_net_info(path_to_params):
    vgg_data = scipy.io.loadmat(path_to_params)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0,1)) # 평균
    network_weights = vgg_data['layers'][0] # 가중치
    return(mat_mean, network_weights) # 수치의평균과 network_weights가중치
    

# Create the VGG-19 Network
# 가중치 및 계층 정의로부터 tf신경망을 재구축해주는 함수
def vgg_network(network_weights, init_image): # 
    # 매개변수
    # network_weights: 사전학습망에서 구한 가중치 정보
    # init_image: 원본 이미지의 placeholder 정보
    network = {} # 딕셔너리 
    image = init_image

    for i, layer in enumerate(vgg_layers): #36번 돈다.
        if layer[0] == 'c':
            weights, bias = network_weights[i][0][0][0][0]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
            image = tf.nn.bias_add(conv_layer, bias)
        elif layer[0] == 'r':
            image = tf.nn.relu(image)
        else:
            image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        network[layer] = image
    
    #네트워크는 다음과 같은 형식으로 데이터가 들어있을 것이다.
    # network = {'conv1_1':'cnn수행결과1', 'relu1_1':'cnn수행결과2', ...}  해당이미지에 대한 정보가 딕셔너리형태로 들어있음. 이렇게 순차적으로 들어있거나 문자로 들어있는것은 아님.
    
    return(network)

# Here we define which layers apply to the original or style image
# 원본이미지에 적용시킬 레이어
original_layer = 'relu4_2'

# 스타일 이미지에 적용시킬 레이어
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# Get network parameters
# normalization_mean : 평균
# network_weights: 가중치
normalization_mean, network_weights = extract_net_info(vgg_path)

# read하여 original image shape가 3인데
# tf의 이미지 연산은 4차원이다.
# 앞에 차원 1개 추가하기 위하여 (1, )을 붙인다.
shape = (1,) + original_image.shape 
style_shape = (1,) + style_image.shape
original_features = {}
style_features = {}

# Get network parameters
image = tf.placeholder('float', shape=shape)
vgg_net = vgg_network(network_weights, image)

# Normalize original image
# 이미지를 정규화하고, 신경망을 통해 실행시킨다.
original_minus_mean = original_image - normalization_mean
original_norm = np.array([original_minus_mean])
original_features[original_layer] = sess.run(vgg_net[original_layer],
                                             feed_dict={image: original_norm})

# Get style image network
# 스타일 이미지를 위한 플래이스 홀더 지정
image = tf.placeholder('float', shape=style_shape)
vgg_net = vgg_network(network_weights, image)

# 스타일 이미지에 대한 정규화
style_minus_mean = style_image - normalization_mean
style_norm = np.array([style_minus_mean])

# 신경망 실행
for layer in style_layers:
    layer_output = sess.run(vgg_net[layer], feed_dict={image: style_norm})
    layer_output = np.reshape(layer_output, (-1, layer_output.shape[3]))
    style_gram_matrix = np.matmul(layer_output.T, layer_output) / layer_output.size
    style_features[layer] = style_gram_matrix

# Make Combined Image
# 결합 이미지를 생성하기 위하여 잡음 이미지를 신경망에 투입하여 실행하다.
# shape: 원본 이미지의 shape
initial = tf.random_normal(shape) * 0.256
image = tf.Variable(initial)
vgg_net = vgg_network(network_weights, image)

# Loss(쿡북 391)
original_loss = original_image_weight * (2 * tf.nn.l2_loss(vgg_net[original_layer] - original_features[original_layer]) /
                original_features[original_layer].size)
                
# Loss from Style Image
# 각각의 스타일 이미지에 대한 비용 계산
style_loss = 0
style_losses = []

# style_layers: 스타일 이미지에 적용시킬 계층정보 
for style_layer in style_layers:
    
    # vgg_net[style_layer]: 연산이 이미 되어 있는(rank:4)
    layer = vgg_net[style_layer]
    feats, height, width, channels = [x.value for x in layer.get_shape()]
    size = height * width * channels
    features = tf.reshape(layer, (-1, channels))
    style_gram_matrix = tf.matmul(tf.transpose(features), features) / size
    style_expected = style_features[style_layer]
    style_losses.append(2 * tf.nn.l2_loss(style_gram_matrix - style_expected) / style_expected.size)

# style_image_weight: 스타일 이미지의 가중치
style_loss += style_image_weight * tf.reduce_sum(style_losses)
        
# To Smooth the resuts, we add in total variation loss       
total_var_x = sess.run(tf.reduce_prod(image[:,1:,:,:].get_shape()))
total_var_y = sess.run(tf.reduce_prod(image[:,:,1:,:].get_shape()))
first_term = regularization_weight * 2
second_term_numerator = tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:])
second_term = second_term_numerator / total_var_y
third_term = (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) / total_var_x)

# total_variation_loss: 전체 변이 비용
# 쿡북 392쪽
# 깨끗한 이미지는 변이 값이 낮고
# 잡음이 많은 이미지는 변이 값이 높다.
total_variation_loss = first_term * (second_term + third_term)
    
# Combined Loss
# 손실 = 원본계층비용 + 스타일계층비용 + 전체 변이 비용
loss = original_loss + style_loss + total_variation_loss

# Declare Optimization Algorithm
optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2) 
#매개변수가 두개, 아담옵티마이저가 모멘텀 관성이 있다.
train_step = optimizer.minimize(loss)

# Initialize Variables and start Training
sess.run(tf.global_variables_initializer())
for i in range(generations): # 
    
    sess.run(train_step)

    # Print update and save temporary output
    if (i+1) % output_generations == 0:
        print('Generation {} out of {}, loss: {}'.format(i + 1, generations,sess.run(loss)))
        image_eval = sess.run(image)
        best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
        
        # 중간과정의 임시 이미지 저장하기
        output_file = 'temp_output_{}.jpg'.format(i)
        scipy.misc.imsave(output_file, best_image_add_mean)
        
        
# Save final image
# 최종 이미지 파일
image_eval = sess.run(image)
best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
output_file = 'final_output.jpg'
scipy.misc.imsave(output_file, best_image_add_mean)
#scientific python 


# 2018-01-18 18:44:33.494552: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
# Generation 20 out of 1000, loss: 273768576.0
# Generation 40 out of 1000, loss: 273720608.0
# Generation 60 out of 1000, loss: 273660064.0
# Generation 80 out of 1000, loss: 273590144.0
# Generation 100 out of 1000, loss: 273516704.0
# Generation 120 out of 1000, loss: 273443264.0
# Generation 140 out of 1000, loss: 273370592.0
# Generation 160 out of 1000, loss: 273298880.0
# Generation 180 out of 1000, loss: 273227936.0
# Generation 200 out of 1000, loss: 273156960.0
# Generation 220 out of 1000, loss: 273085984.0
# Generation 240 out of 1000, loss: 273014400.0
# Generation 260 out of 1000, loss: 272942048.0
# Generation 280 out of 1000, loss: 272868736.0
# Generation 300 out of 1000, loss: 272794112.0
# Generation 320 out of 1000, loss: 272718624.0
# Generation 340 out of 1000, loss: 272641440.0
# Generation 360 out of 1000, loss: 272562624.0
# Generation 380 out of 1000, loss: 272482400.0
# Generation 400 out of 1000, loss: 272400320.0
# Generation 420 out of 1000, loss: 272316352.0
# Generation 440 out of 1000, loss: 272230528.0
# Generation 460 out of 1000, loss: 272142592.0
# Generation 480 out of 1000, loss: 272052320.0
# Generation 500 out of 1000, loss: 271959872.0
# Generation 520 out of 1000, loss: 271865120.0
# Generation 540 out of 1000, loss: 271768096.0
# Generation 560 out of 1000, loss: 271668576.0
# Generation 580 out of 1000, loss: 271566720.0
# Generation 600 out of 1000, loss: 271462272.0
# Generation 620 out of 1000, loss: 271355296.0
# Generation 640 out of 1000, loss: 271245632.0
# Generation 660 out of 1000, loss: 271133184.0
# Generation 680 out of 1000, loss: 271018304.0
# Generation 700 out of 1000, loss: 270900512.0
# Generation 720 out of 1000, loss: 270779968.0
# Generation 740 out of 1000, loss: 270656736.0
# Generation 760 out of 1000, loss: 270530848.0
# Generation 780 out of 1000, loss: 270402496.0
# Generation 800 out of 1000, loss: 270271520.0
# Generation 820 out of 1000, loss: 270137664.0
# Generation 840 out of 1000, loss: 270001152.0
# Generation 860 out of 1000, loss: 269861600.0
# Generation 880 out of 1000, loss: 269719488.0
# Generation 900 out of 1000, loss: 269574656.0
# Generation 920 out of 1000, loss: 269426880.0
# Generation 940 out of 1000, loss: 269276224.0
# Generation 960 out of 1000, loss: 269123392.0
# Generation 980 out of 1000, loss: 268967360.0
# Generation 1000 out of 1000, loss: 268808704.0
