# chap_02_01_Data_Graph.py 
#계산 그래프의 연산(쿡북64)
#placeholder를 사용한 계산 그래프의 연산
#구구단3단
# Operations on a Computational Graph
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.]) #입력할데이터셋
#x_vals = np.array([1., 3., 5., 7., 9.]) #입력할데이터셋 
x = np.linspace(1, 9, num=5)
print(x)
x = np.linspace(1, 9, num=9)
print(x)
x_data = tf.placeholder(tf.float32)#입력을 위한 placeholder
m = tf.constant(3.) #단수:3단
print('단수3단:',m)
m = tf.constant(4.) 
print('단수4단:',m)
# Multiplication
prod = tf.multiply(x_data, m) #어떤수*3
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))

merged = tf.summary.merge_all()
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)