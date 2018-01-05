import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



x = [1.0]
y = [1.0]

w = tf.placeholder(tf.float32)

H = x * w

cost = tf.reduce_mean(tf.square(H-y))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

mylist = []
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, w], feed_dict={w: feed_W})
    if i % 5 == 0 :
        sublist = []
        sublist.append(curr_W)
        sublist.append(curr_cost)
        mylist.append( sublist )
        
    W_val.append(curr_W)
    mylist.append( sublist )
    
for item in mylist:
    print(item)
    
plt.plot(W_val, cost_val)
plt.show()


