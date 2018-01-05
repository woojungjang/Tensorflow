import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow1111.p40matrix import x_column

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

def col_length( input ):
    sublist = input[0]
    length = len(sublist)
    return length

x_column = col_length(x_data)
y_column = col_length(y_data)

weight1_row = x_column
weight1_column = 2
bias1 = weight1_column

w1 = tf.Variable(tf.random_normal([weight1_row, weight1_column]))
b1 = tf.Variable(tf.random_normal([bias1]))
layer1 = tf.sigmoid(tf.matmul(x, w1)+b1)

weight2_row = weight1_column
weight2_column = y_column
bias2 = weight2_column

w2= tf.Variable(tf.random_normal([weight2_row, weight2_column]))
b2=tf.Variable(tf.random_normal([bias2]))
H = tf.sigmoid(tf.matmul(layer1, w2)+b2)
diff = y * tf.log(H) + (1-y)*tf.log(1-H)
cost = -tf.reduce_mean(diff)

learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
train = optimizer.minimize(cost)
predicted = tf.cast(H>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={x:x_data, y:y_data})
        if step % 1000 == 0:
            _cost, _w = sess.run([cost, w], feed_dict={x:x_data, y:y_data})
            print('step:', step,',cost:',_cost,',weight:', _w)
    hypothesis, _predicted, _accuracy = sess.run([H, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print('hypothesis:', hypothesis, 'Correct:',_predicted, 'accuracy', _accuracy)

