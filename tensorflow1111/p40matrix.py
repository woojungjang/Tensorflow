import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Linearregression20171222 import optimizer

A = np.array([[1, 2]])
B = np.array([[-2], [3]])
C = np.array([[3, 2], [-1,0]])

print(A.dot(B))
print(A.dot(C))
print(B.dot(A))
print(C.dot(B))

############################

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[215.], [555.], [555.], [666.], [555.]]

x_column = 3
y_column = 1

x = tf.placeholder(tf.float32, shape=[None, x_column])
y = tf.placeholder(tf.float32, shape=[None, y_column])

w = tf.Variable( tf.random_normal([x_column, y_column]))
b = tf.Variable( tf.random_normal([1]))

H = tf.matmul(x, w) + b
diff = tf.square( H - y )
cost = tf.reduce_mean( diff )

learn_rate = 1e-9
optimizer = tf.train.GradientDescentOptimizer( learning_rate = learn_rate )
train = optimizer.minimize( cost )

sess = tf.Session()
sess.run( tf.global_variables_initializer())

for step in range( 700001):
    _c, _h, _t = sess.run([cost, H, train], feed_dict={x: x_data, y : y_data})
    if step % 500 == 0 :
        print("step:", step, "\nCost:", _c, "\nPrediction:\n", _h)
        print('---------------------')