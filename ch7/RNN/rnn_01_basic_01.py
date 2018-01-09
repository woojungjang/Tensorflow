# rnn_01_basic_01.py
 
# 예시는 4개의 입력(input-dim)이 들어가서 2(hidden_size)개의 출력이 나오는 예시이다.
# input_dim : 4개
# hidden_size : 2개
 
import tensorflow as tf
import numpy as np

hidden_size = 512

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[[1,0,0,0]]], dtype=np.float32) 

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(outputs))