# rnn_01_basic_03.py
# 문자열을 여러 셋트(예시에서는 3 세트) 넣어서 일괄 처리해준다.  
 
import tensorflow as tf
import numpy as np

hidden_size = 2

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[[ 1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]],
    
    [[0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.]],
    
    [[0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.]]], dtype=np.float32) 

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(outputs))