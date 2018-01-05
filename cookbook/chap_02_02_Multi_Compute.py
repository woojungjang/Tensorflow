#chap_02_02_Multi_Compute.py(쿡북66-67)
# p67
# prod1: #(3*5)'(5*1) => (3*1)
# prod2: #(3*1)'(상수) => (3*1)
# add1: #(3*1)'(상수) => (3*1)

import tensorflow as tf
import numpy as np
from neuralnet.mnist import batch_size
sess = tf.Session()

my_array = np.array([[1.,3.,5.,7.,9.],[-2.,0.,2.,4.,6.],[-6.,-3.,0.,3.,6.]])
x_vals = np.array([my_array,my_array+1])
x_data = tf.placeholder(tf.float32, shape=(3,5))
m1 = tf.constant([[1.],[0.],[-1],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))
x_data = tf.placeholder(tf.float32, shape=(3,None))

import os
import io
import time

summary_writer = tf.summary.FileWriter('tensorboard',tf.get_default_graph())
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
    
    batch_size = 503
    