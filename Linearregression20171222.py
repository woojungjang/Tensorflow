import tensorflow as tf
import numpy as np

x = np.arange( 1.0, 4.1, 5.1 )
print( 'x = ',x)
y = [3.1, 4.1, 4.9, 6.1, 6.1, 6.7, 5.4]
print( 'y = ', y)

w = tf.Variable(0.0001)
b = tf.Variable(0.00001)

H = w*x + b

diff = tf.square(H - y)
cost = tf.reduce_mean(diff)

learn_rate = 1e-3
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
weight_list = []

for step in range(10000):
    sess.run(train)
    cost_list.append(sess.run(cost))
    weight_list.append(sess.run(w))

#import matplotlib.pyplot as plt



#    print('step : %d, cost : %.12f, weight : %f, bias : %f' %\
#        (step, sess.run(cost), sess.run(w), sess.run(b)))
    


#x = [8. 1.5 2. 2.5 3. 3.5 4.]
#y = [3.1, 4.1, 4.9, 6.1, 3.4, 5.1, 4.1]