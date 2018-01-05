import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



# input place holders
X = tf.placeholder(tf.float32, [None, 100, 80, 3])
Y = tf.placeholder(tf.float32, [None, 100])
 
# Layer 1 input shape=(?, 100, 80, 3)
W1 = tf.Variable(tf.random_normal([7, 3, 3, 10], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
# Conv -&gt; (?, 100, 78, 10)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')
# Pool -&gt; (?, 100, 26, 10)
 
# Layer 2 input shape=(?, 100, 26, 10)
W2 = tf.Variable(tf.random_normal([3, 3, 10, 20], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# Conv -&gt; (?, 100, 24, 20)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')
# Pool -&gt; (?, 100, 8, 20)
 
# Flattening (except for time-axis)
L2_flat = tf.reshape(L2, [-1, 100, 8 * 20])
print('L2_flat after flattening: {}'.format(L2_flat))