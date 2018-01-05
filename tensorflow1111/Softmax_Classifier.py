# softMaxEx01.py
# 4개의 입력 정보를 이용하여 3개의 클래스 중 1개로 분류해주는 예시
 
import tensorflow as tf
import numpy as np
 
# 중첩 리스트의 열 갯수 구하는 함수
def col_length( input ):
    sublist = input[0]
    length = len(sublist)
    return length
 
def getCategory( datalist ):
    mylist = ['강아지', '고양이', '토끼']
 
    for item in range(len(datalist)):
        print(datalist[item], mylist[(int)(datalist[item])])
 
# x_data : 2행 4열
x_data = [[1, 2, 1, 1], [2, 1, 3, 2]]
 
# y_data : 2행 3열
y_data = [[0, 0, 1], [0, 0, 1]]
 
column = col_length( x_data)
print( '컬럼수 : ', column )
nb_classes = col_length( y_data )
print( '클래스 갯수 : ', nb_classes )
 
# abc행 column(4)열
x = tf.placeholder("float", [None, column])
 
# abc행 nb_classes(3)열
y = tf.placeholder("float", [None, nb_classes])
 
# w : 4행 3열
w = tf.Variable(tf.random_normal([column, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
 
# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# (abc행 4열) * (4행 3열) ==> (abc행 3열)
H = tf.nn.softmax(tf.matmul(x, w) + b)
 
# Cross entropy cost/loss
diff = -tf.reduce_sum(y * tf.log(H), axis=1)
cost = tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize( cost )
 
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
 
   for step in range(2001):
       sess.run(train, feed_dict={x: x_data, y: y_data})
       if step % 500 == 0:
           _c, _h = sess.run([cost, H], feed_dict={x: x_data, y: y_data})
           print('학습 회수 :', step, ', 비용 :',_c)
           print('가설 :\n', _h)
           print('최대값 인덱스(argmax) :', np.argmax(_h, axis=1))
           print('최종 결과:')
           getCategory(np.argmax(_h, axis=1))
           print('--------------------------------------')