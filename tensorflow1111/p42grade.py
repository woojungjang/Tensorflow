import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


'''x_data = [[10.0, 20, 30], [100, 90, 80], [50, 55, 45]]
y_data = [[40],[70],[50]]
x_test = [[20,40,50],[90,88,80]]

x_column = 3
y_column = 1
x = tf.placeholder(tf.float32, shape=[None, x_column])
y = tf.placeholder(tf.float32, shape=[None, y_column])

w = tf.Variable(tf.random_normal([x_column, y_column]))
b = tf.Variable(0,0)

H = tf.matmul(x, w) + b

diff = tf.square( H - y )
cost = tf.reduce_mean( diff )

learn_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer( learning_rate=learn_rate)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run( tf.global_variables_initializer())
for step in range(1000):
    _t, _w, _c, _h = sess.run([train, w, cost, H], feed_dict={ x: x_data, y:y_data})
    if step % 10 == 0:
        print( 'step: %d, cost : %f' % ( step, _c ))'''
        
# matrixOperator02.py
import tensorflow as tf
 
#성적 변화, 다음번 성적 예측
# x_data : shape(3, 3)
x_data = [[10.0, 20, 30],[100, 90, 80],[50, 55, 45]]
# y_data : shape(3, 1)
y_data = [[40], [70], [50]]
# x_test : shape(2, 3)
x_test =  [[20, 40, 50],[90, 88, 80]]
 
# 3 : 특징의 갯수, None : m(전체 3명의 데이터)
x_column = 3 # 입력 데이터의 컬럼 갯수
y_column = 1
x = tf.placeholder( tf.float32, shape=[None, x_column])
y = tf.placeholder( tf.float32, shape=[None, y_column])
 
# 데이터가 3개인데, 3만 쓰면 안되고 n by m의 형식으로 3행 1열로 만들었다.
w = tf.Variable(tf.random_normal([x_column, y_column]))
b = tf.Variable(0.0)
 
H = tf.matmul(x, w) + b
 
diff = tf.square( H- y )
cost = tf.reduce_mean( diff )
 
learn_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer( learning_rate = learn_rate )
train = optimizer.minimize( cost )
 
sess = tf.Session()
sess.run( tf.global_variables_initializer() ) # 변수 초기화
 
for step in range(10000) :
    _t, _w, _c, _h = sess.run([train, w, cost, H], feed_dict={ x : x_data, y : y_data })
    if step % 10 == 0 :
        print( 'step : %d, cost : %f' % ( step, _c ) )
 
# x_test : 2행, 3열이다.
result = sess.run(H, feed_dict={ x : x_test })
print('score predict : ', result )
