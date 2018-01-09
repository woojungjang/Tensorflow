# rnn_01_basic_02.py
# 알파벳 5개(sequence_length=5)를 입력해주겠다. 
 
import tensorflow as tf
import numpy as np

hidden_size = 2

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[[ 1.,  0.,  0.,  0.],
    [ 0.,  1.,  0.,  0.],
    [ 0.,  0.,  1.,  0.],
    [ 0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.]]], dtype=np.float32) 

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(outputs))